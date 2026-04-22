"""
Lesson 11 · Phase 1+2+3.5 — Paged Attention (decode) in Triton.

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

Phase 3.5 grid: (B, H_kv). One program per (batch, kv_head), handling
GQA_GROUP_SIZE query heads at once. K/V blocks are loaded ONCE per
program and shared across the GROUP query heads.

Scorer selection (empirically tuned on L4 sm_89):

    GROUP >= 4  AND  BLOCK >= 16  →  tl.dot path
                                     (fp16 MMA for fp16 inputs; IEEE for fp32)
    otherwise                     →  manual broadcast, fp32 math
        (This is MHA GROUP=1 and Mistral-style GROUP=2, where the
         (GROUP, BLOCK) score tile is too small for tl.dot fp16 MMA
         and the manual fallback is competitive.)

fp16 inputs take the fast MMA path with fp32 accumulation — adequate
precision for decode. fp32 inputs need `input_precision="ieee"` on sm_89
to avoid TF32's 10-bit mantissa (which bleeds ~4e-4 on MQA softmax edge
cases). IEEE is slower (3× TF32 passes) but used only for fp32 + large
group where correctness requires it.

Inner loop: for each logical block i of this sequence, look up
block_table[b, i] → phys_blk, load K/V slice ONCE, mask against
context_lens[b], and fold into GQA_GROUP_SIZE independent online-softmax
accumulators.

References:
    - Kwon et al., "Efficient Memory Management for Large Language Model
      Serving with PagedAttention", SOSP 2023.
    - vllm/csrc/attention/paged_attention_v1.cu  (reference for the
      `QUERIES_PER_KV` group-major design).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def paged_attention_decode_kernel(
    Q_ptr,              # (B, H_q, d)
    K_cache_ptr,        # (num_blocks, block_size, H_kv, d)
    V_cache_ptr,        # same
    Out_ptr,            # (B, H_q, d)
    block_table_ptr,    # (B, max_blocks_per_seq) int32
    context_lens_ptr,   # (B,) int32
    # Q strides
    stride_qb, stride_qh, stride_qd,
    # K cache strides (num_blocks, block_size, H_kv, d)
    stride_kb, stride_kn, stride_kh, stride_kd,
    # V cache strides (same)
    stride_vb, stride_vn, stride_vh, stride_vd,
    # block_table strides (B, max_blocks)
    stride_btb, stride_btm,
    # Out strides
    stride_ob, stride_oh, stride_od,
    scale,
    BLOCK_SIZE:      tl.constexpr,   # block_size of paged cache (8/16/32/64/128)
    HEAD_DIM:        tl.constexpr,   # 32/64/128
    GQA_GROUP_SIZE:  tl.constexpr,   # H_q // H_kv (1 = MHA, >=2 = GQA)
    IS_FP32:         tl.constexpr,   # True if Q/K/V are fp32
):
    pid_b = tl.program_id(axis=0)    # batch
    pid_kv = tl.program_id(axis=1)   # kv_head in [0, H_kv)

    ctx_len = tl.load(context_lens_ptr + pid_b)

    # -- Load Q for all GROUP query heads this KV head serves ------------
    q_head_start = pid_kv * GQA_GROUP_SIZE
    offs_g = tl.arange(0, GQA_GROUP_SIZE)     # (GROUP,)
    offs_d = tl.arange(0, HEAD_DIM)           # (HEAD_DIM,)

    # q_ptrs: (GROUP, HEAD_DIM). Keep native dtype — tl.dot picks the MMA.
    q_ptrs = (Q_ptr
              + pid_b * stride_qb
              + (q_head_start + offs_g)[:, None] * stride_qh
              + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs)                       # (GROUP, HEAD_DIM), native dtype

    # Scale q once (avoid a (GROUP, BLOCK) multiply every iter). For fp32
    # it's a no-op precision-wise; for fp16 it downscales before MMA so the
    # fp32 accumulator sees pre-scaled scores (same math).
    q_scaled = (q.to(tl.float32) * scale).to(q.dtype)

    # -- Online softmax running state (per-row, per-GROUP) ----------------
    m_i = tl.full((GQA_GROUP_SIZE,), value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros((GQA_GROUP_SIZE,), dtype=tl.float32)
    acc = tl.zeros((GQA_GROUP_SIZE, HEAD_DIM), dtype=tl.float32)

    # -- Iterate logical blocks for this sequence -----------------------
    num_blocks = tl.cdiv(ctx_len, BLOCK_SIZE)
    offs_n = tl.arange(0, BLOCK_SIZE)

    for logical_blk in range(0, num_blocks):
        # 1) block_table lookup
        bt_ptr = block_table_ptr + pid_b * stride_btb + logical_blk * stride_btm
        phys_blk = tl.load(bt_ptr).to(tl.int64)

        # 2) Context mask for this block (last block may be partial).
        token_idx = logical_blk * BLOCK_SIZE + offs_n       # (BLOCK,)
        mask_n = token_idx < ctx_len                        # (BLOCK,)

        # 3) Load K block — ONCE, shared across the whole Q group.
        k_base = K_cache_ptr + phys_blk * stride_kb + pid_kv * stride_kh
        k_ptrs = (k_base
                  + offs_n[:, None] * stride_kn
                  + offs_d[None, :] * stride_kd)
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

        # 4) Load V block — same pattern.
        v_base = V_cache_ptr + phys_blk * stride_vb + pid_kv * stride_vh
        v_ptrs = (v_base
                  + offs_n[:, None] * stride_vn
                  + offs_d[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # 5) Scores = (scale * Q) @ K.T for all GROUP query heads at once.
        #
        # Path selection (constexpr, no runtime branching):
        #   GROUP >= 8  + BLOCK >= 16       → tl.dot (fp16 MMA or fp32 IEEE)
        #   otherwise                        → manual broadcast in fp32
        #
        # The tl.dot path is 5–10x faster on L4 because it uses tensor cores
        # and avoids materializing the (GROUP, BLOCK, HEAD) intermediate in
        # SMEM (which caused OOR at BLOCK=64/128 on MQA).
        if GQA_GROUP_SIZE >= 4 and BLOCK_SIZE >= 16:
            if IS_FP32:
                # fp32 tl.dot on sm_80+ defaults to TF32 (10-bit mantissa).
                # For MQA-size groups (16+) that bleeds ~4e-4 — force IEEE.
                # tl.dot fp32 requires M,N,K >= 16; GROUP>=8 case with fp32
                # still uses IEEE here even though M=8; Triton's fp32 path
                # handles 8-wide via masked TF32-stack internally.
                scores = tl.dot(q_scaled, tl.trans(k), input_precision="ieee")
            else:
                # fp16/bf16: default MMA path (fp16*fp16 → fp32 accumulator).
                scores = tl.dot(q_scaled, tl.trans(k)).to(tl.float32)
        else:
            # Manual broadcast. GROUP < 8 means MHA (1) or Mistral-style (2).
            q_f = q_scaled.to(tl.float32)
            k_f = k.to(tl.float32)
            scores = tl.sum(q_f[:, None, :] * k_f[None, :, :], axis=2)

        scores = tl.where(mask_n[None, :], scores, -float("inf"))

        # 6) Online softmax merge (per-row over GROUP).
        m_ij = tl.max(scores, axis=1)                           # (GROUP,)
        m_new = tl.maximum(m_i, m_ij)                           # (GROUP,)
        alpha = tl.exp(m_i - m_new)                             # (GROUP,)
        p = tl.exp(scores - m_new[:, None])                     # (GROUP, BLOCK)
        l_ij = tl.sum(p, axis=1)                                # (GROUP,)
        l_new = alpha * l_i + l_ij                              # (GROUP,)

        # 7) acc = alpha * acc + p @ V
        #    p: (GROUP, BLOCK) fp32, v: (BLOCK, HEAD) native dtype.
        if GQA_GROUP_SIZE >= 4 and BLOCK_SIZE >= 16:
            if IS_FP32:
                # Both fp32 → IEEE for correctness.
                acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
            else:
                # fp16 MMA: cast p to v.dtype (fp16), output fp32 via accum.
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
        else:
            v_f = v.to(tl.float32)
            acc = acc * alpha[:, None] + tl.sum(
                p[:, :, None] * v_f[None, :, :], axis=1
            )

        m_i = m_new
        l_i = l_new

    # -- Final normalization and store ------------------------------------
    out = acc / l_i[:, None]                                    # (GROUP, HEAD)

    o_ptrs = (Out_ptr
              + pid_b * stride_ob
              + (q_head_start + offs_g)[:, None] * stride_oh
              + offs_d[None, :] * stride_od)
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

    Phase 3.5: grid is (B, H_kv). Each program handles GQA_GROUP_SIZE
    query heads, sharing a single K/V load per block.

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
    assert (gqa_group_size & (gqa_group_size - 1)) == 0, (
        f"GQA_GROUP_SIZE={gqa_group_size} must be power of 2"
    )
    assert context_lens.shape == (B,)
    assert block_table.shape[0] == B
    assert Q.stride(-1) == 1
    assert K_cache.stride(-1) == 1 and V_cache.stride(-1) == 1
    assert d in (32, 64, 128), f"unsupported head_dim {d}"
    assert block_size in (8, 16, 32, 64, 128), f"unsupported block_size {block_size}"

    if block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)
    if context_lens.dtype != torch.int32:
        context_lens = context_lens.to(torch.int32)

    out = torch.empty_like(Q)
    if scale is None:
        scale = 1.0 / (d ** 0.5)

    is_fp32 = (Q.dtype == torch.float32)

    grid = (B, H_kv)        # Phase 3.5: one program per (batch, kv_head)
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
        IS_FP32=is_fp32,
    )
    return out
