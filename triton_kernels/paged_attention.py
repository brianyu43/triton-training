"""
Lesson 11 · Phase 1+2 — Paged Attention (decode) in Triton.

vLLM-style paged KV cache: the cache is stored as a pool of fixed-size
blocks, and each sequence carries a `block_table` that maps its logical
token positions to physical block ids. The attention kernel walks the
block_table per-sequence, loads each block, and runs online softmax.

Scope:
    - Decode only (Q has one token per sequence).
    - MHA (H_q == H_kv) or GQA (H_q % H_kv == 0, each KV head shared by
      GQA_GROUP_SIZE query heads). LLaMA-70B-style grouped-query works.
    - fp16 or fp32, head_dim ∈ {32, 64, 128}.

Shapes:
    Q            : (B, H_q,  d)
    K_cache      : (num_blocks, block_size, H_kv, d)
    V_cache      : (num_blocks, block_size, H_kv, d)
    block_table  : (B, max_blocks_per_seq)   int32
    context_lens : (B,)                      int32
    Out          : (B, H_q,  d)

Grid: (B, H_q). One program per (batch, query_head). The KV head for a
program is kv_head = pid_h // GQA_GROUP_SIZE.

Inner loop: for each logical block i of this sequence, look up
block_table[b, i] → phys_blk, load K_cache[phys_blk, :, kv_head, :] and
V_cache[phys_blk, :, kv_head, :], mask against context_lens[b], and fold
into the online-softmax accumulator.

Notes / deliberate simplifications for Phase 1:
    - Manual dot (`tl.sum(q[None,:] * k, axis=1)`) instead of `tl.dot`.
      block_size is typically 16; tl.dot requires M,N ≥ 16 and we only
      have one query row, so we'd need to pad Q. Defer that to Phase 3.
    - No autotune yet. Hard-coded BLOCK_SIZE matching the runtime value.
    - Single partition per sequence (no FA-v2 style split across programs).
      For very long context (≥16k), vLLM's v2 kernel splits; defer.

References:
    - Kwon et al., "Efficient Memory Management for Large Language Model
      Serving with PagedAttention", SOSP 2023.
    - vllm/csrc/attention/paged_attention_v1.cu
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def paged_attention_decode_kernel(
    Q_ptr,              # (B, H, d)
    K_cache_ptr,        # (num_blocks, block_size, H, d)
    V_cache_ptr,        # same
    Out_ptr,            # (B, H, d)
    block_table_ptr,    # (B, max_blocks_per_seq) int32
    context_lens_ptr,   # (B,) int32
    # Q strides
    stride_qb, stride_qh, stride_qd,
    # K cache strides (num_blocks, block_size, H, d)
    stride_kb, stride_kn, stride_kh, stride_kd,
    # V cache strides (same)
    stride_vb, stride_vn, stride_vh, stride_vd,
    # block_table strides (B, max_blocks)
    stride_btb, stride_btm,
    # Out strides
    stride_ob, stride_oh, stride_od,
    scale,
    BLOCK_SIZE:      tl.constexpr,   # block_size of paged cache (16/32/64)
    HEAD_DIM:        tl.constexpr,   # 32/64/128
    GQA_GROUP_SIZE:  tl.constexpr,   # H_q // H_kv (1 = MHA, >1 = GQA)
):
    pid_b = tl.program_id(axis=0)   # batch
    pid_h = tl.program_id(axis=1)   # query head in [0, H_q)

    # KV head this query head reads from. MHA → GQA_GROUP_SIZE=1 → kv_head=pid_h.
    kv_head = pid_h // GQA_GROUP_SIZE

    ctx_len = tl.load(context_lens_ptr + pid_b)

    # -- Load Q (single vector, shape (HEAD_DIM,)) -------------------------
    offs_d = tl.arange(0, HEAD_DIM)
    q_ptrs = Q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_d * stride_qd
    q = tl.load(q_ptrs)   # fp16 or fp32

    # Promote to fp32 for stable softmax.
    q = q.to(tl.float32)

    # -- Online softmax running state --------------------------------------
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    # -- Iterate logical blocks for this sequence -------------------------
    # num_blocks = ceil(ctx_len / BLOCK_SIZE). We compute it as a host-time
    # upper bound inside the kernel via tl.cdiv.
    num_blocks = tl.cdiv(ctx_len, BLOCK_SIZE)
    offs_n = tl.arange(0, BLOCK_SIZE)

    for logical_blk in range(0, num_blocks):
        # 1) block_table lookup
        bt_ptr = block_table_ptr + pid_b * stride_btb + logical_blk * stride_btm
        phys_blk = tl.load(bt_ptr).to(tl.int64)

        # 2) Context mask for this block (last block may be partial).
        token_idx = logical_blk * BLOCK_SIZE + offs_n
        mask_n = token_idx < ctx_len

        # 3) Load K block — shape (BLOCK_SIZE, HEAD_DIM), use kv_head for GQA.
        k_base = K_cache_ptr + phys_blk * stride_kb + kv_head * stride_kh
        k_ptrs = (k_base
                  + offs_n[:, None] * stride_kn
                  + offs_d[None, :] * stride_kd)
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        # 4) Load V block — same KV head.
        v_base = V_cache_ptr + phys_blk * stride_vb + kv_head * stride_vh
        v_ptrs = (v_base
                  + offs_n[:, None] * stride_vn
                  + offs_d[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        # 5) Scores s = (q · k_row) * scale for each of BLOCK_SIZE rows.
        # Broadcast-multiply + reduce instead of tl.dot — fine for small
        # BLOCK_SIZE and a single query row.
        scores = tl.sum(q[None, :] * k, axis=1) * scale      # (BLOCK_SIZE,)
        scores = tl.where(mask_n, scores, -float("inf"))

        # 6) Online softmax merge.
        m_ij = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)                             # (BLOCK_SIZE,)
        l_ij = tl.sum(p, axis=0)
        l_new = alpha * l_i + l_ij

        # 7) Accumulate output: acc = alpha*acc + p @ V
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)     # (HEAD_DIM,)

        m_i = m_new
        l_i = l_new

    # -- Final normalization and store ------------------------------------
    out = acc / l_i
    o_ptrs = Out_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_d * stride_od
    tl.store(o_ptrs, out.to(Out_ptr.dtype.element_ty))


def triton_paged_attention_decode(
    Q: torch.Tensor,
    K_cache: torch.Tensor,
    V_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Paged attention forward (decode, MHA or GQA).

    Args:
        Q: (B, H_q, d) — last axis contiguous.
        K_cache, V_cache: (num_blocks, block_size, H_kv, d) — last axis contig.
            H_q must be divisible by H_kv. GQA_GROUP_SIZE = H_q // H_kv.
        block_table: (B, max_blocks_per_seq) int32.
        context_lens: (B,) int32.
        scale: default 1/sqrt(d).

    Returns:
        (B, H_q, d) attention output, same dtype as Q.
    """
    assert Q.is_cuda and K_cache.is_cuda and V_cache.is_cuda
    assert Q.dim() == 3 and K_cache.dim() == 4 and V_cache.dim() == 4
    assert K_cache.shape == V_cache.shape
    assert block_table.dim() == 2 and context_lens.dim() == 1
    assert block_table.dtype in (torch.int32, torch.int64)
    assert context_lens.dtype in (torch.int32, torch.int64)

    B, H_q, d = Q.shape
    num_blocks, block_size, H_kv, d_kv = K_cache.shape
    assert H_q % H_kv == 0, f"H_q={H_q} must be divisible by H_kv={H_kv} (GQA)"
    assert d == d_kv
    gqa_group_size = H_q // H_kv
    assert context_lens.shape == (B,)
    assert block_table.shape[0] == B
    assert Q.stride(-1) == 1
    assert K_cache.stride(-1) == 1 and V_cache.stride(-1) == 1
    assert d in (32, 64, 128), f"unsupported head_dim {d}"
    assert block_size in (8, 16, 32, 64, 128), f"unsupported block_size {block_size}"

    # Coerce index tensors to int32 to match kernel expectation.
    if block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)
    if context_lens.dtype != torch.int32:
        context_lens = context_lens.to(torch.int32)

    out = torch.empty_like(Q)
    if scale is None:
        scale = 1.0 / (d ** 0.5)

    grid = (B, H_q)
    paged_attention_decode_kernel[grid](
        Q, K_cache, V_cache, out,
        block_table, context_lens,
        Q.stride(0),         Q.stride(1),         Q.stride(2),
        K_cache.stride(0),   K_cache.stride(1),   K_cache.stride(2),   K_cache.stride(3),
        V_cache.stride(0),   V_cache.stride(1),   V_cache.stride(2),   V_cache.stride(3),
        block_table.stride(0), block_table.stride(1),
        out.stride(0),       out.stride(1),       out.stride(2),
        scale,
        BLOCK_SIZE=block_size,
        HEAD_DIM=d,
        GQA_GROUP_SIZE=gqa_group_size,
    )
    return out
