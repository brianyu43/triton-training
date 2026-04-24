"""
Lesson 11 · Phase 1+2+3.5 + Lesson 12 · Split-K — Paged Attention (decode) in Triton.

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

Two execution paths:

  (a) Single-pass  — grid = (B, H_kv)
        One program per (batch, kv_head) walks every block in the
        sequence and emits a fully-normalized row. Used when the base
        grid is large enough to saturate the GPU.

  (b) Split-K (lesson 12) — grid = (B, H_kv, SEGMENTS) + a reduce kernel.
        Each program walks only PARTITION_SIZE tokens of its sequence,
        writes an UNNORMALIZED (m, l, acc) triple to scratch, and a
        second kernel over (B, H_q) recombines them with the standard
        online-softmax rescale. This rescues shapes whose base grid is
        tiny (e.g. MQA B=16 H_kv=1 → 16 programs on 58 SMs before, 128
        programs after split-k with PARTITION=512 on ctx=4k).

Phase 3.5 grid choice (`(B, H_kv)` rather than `(B, H_q)`): each program
handles GQA_GROUP_SIZE query heads at once, loads K/V blocks ONCE, and
runs GROUP_SIZE parallel online-softmax accumulators. Score and PV paths
use `tl.dot` when `GROUP >= 4 and BLOCK >= 16` (fp16 MMA on sm_89);
everything else falls back to manual broadcast in fp32. fp32 inputs on
the `tl.dot` path use `input_precision="ieee"` to dodge TF32's 10-bit
mantissa, which silently injects ~4e-4 error on MQA softmax edge cases.

References:
    - Kwon et al., "Efficient Memory Management for Large Language Model
      Serving with PagedAttention", SOSP 2023.
    - vllm/csrc/attention/paged_attention_v1.cu  (reference for the
      `QUERIES_PER_KV` group-major design).
    - vllm/csrc/attention/paged_attention_v2.cu  (reference for the
      ctx-axis split-k + reduce pattern; lesson 12).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# =============================================================================
# (a) Single-pass forward kernel.  grid = (B, H_kv).
# =============================================================================

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

    q_ptrs = (Q_ptr
              + pid_b * stride_qb
              + (q_head_start + offs_g)[:, None] * stride_qh
              + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs)                       # (GROUP, HEAD_DIM), native dtype

    q_scaled = (q.to(tl.float32) * scale).to(q.dtype)

    # -- Online softmax running state (per-row, per-GROUP) ----------------
    m_i = tl.full((GQA_GROUP_SIZE,), value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros((GQA_GROUP_SIZE,), dtype=tl.float32)
    acc = tl.zeros((GQA_GROUP_SIZE, HEAD_DIM), dtype=tl.float32)

    num_blocks = tl.cdiv(ctx_len, BLOCK_SIZE)
    offs_n = tl.arange(0, BLOCK_SIZE)

    for logical_blk in range(0, num_blocks):
        bt_ptr = block_table_ptr + pid_b * stride_btb + logical_blk * stride_btm
        phys_blk = tl.load(bt_ptr).to(tl.int64)

        token_idx = logical_blk * BLOCK_SIZE + offs_n
        mask_n = token_idx < ctx_len

        k_base = K_cache_ptr + phys_blk * stride_kb + pid_kv * stride_kh
        k_ptrs = (k_base
                  + offs_n[:, None] * stride_kn
                  + offs_d[None, :] * stride_kd)
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

        v_base = V_cache_ptr + phys_blk * stride_vb + pid_kv * stride_vh
        v_ptrs = (v_base
                  + offs_n[:, None] * stride_vn
                  + offs_d[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        if GQA_GROUP_SIZE >= 4 and BLOCK_SIZE >= 16:
            if IS_FP32:
                scores = tl.dot(q_scaled, tl.trans(k), input_precision="ieee")
            else:
                scores = tl.dot(q_scaled, tl.trans(k)).to(tl.float32)
        else:
            q_f = q_scaled.to(tl.float32)
            k_f = k.to(tl.float32)
            scores = tl.sum(q_f[:, None, :] * k_f[None, :, :], axis=2)

        scores = tl.where(mask_n[None, :], scores, -float("inf"))

        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_ij = tl.sum(p, axis=1)
        l_new = alpha * l_i + l_ij

        if GQA_GROUP_SIZE >= 4 and BLOCK_SIZE >= 16:
            if IS_FP32:
                acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
            else:
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
        else:
            v_f = v.to(tl.float32)
            acc = acc * alpha[:, None] + tl.sum(
                p[:, :, None] * v_f[None, :, :], axis=1
            )

        m_i = m_new
        l_i = l_new

    out = acc / l_i[:, None]

    o_ptrs = (Out_ptr
              + pid_b * stride_ob
              + (q_head_start + offs_g)[:, None] * stride_oh
              + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, out.to(Out_ptr.dtype.element_ty))


# =============================================================================
# (b1) Split-K forward kernel.  grid = (B, H_kv, SEGMENTS).
#
# Each program walks only the blocks inside its PARTITION_SIZE window and
# writes the UNNORMALIZED online-softmax state (m, l, acc) to scratch.
# Segments whose start lies beyond ctx_len run zero iterations — the
# initial state (m=-inf, l=0, acc=0) gets written and the reduce kernel
# treats it as a zero-contribution term via exp(-inf - M_global) = 0.
# =============================================================================

@triton.jit
def paged_attention_split_kernel(
    Q_ptr,
    K_cache_ptr, V_cache_ptr,
    partial_max_ptr,       # (B, H_q, SEGMENTS) fp32
    partial_lse_ptr,       # (B, H_q, SEGMENTS) fp32
    partial_out_ptr,       # (B, H_q, SEGMENTS, d) fp32
    block_table_ptr,
    context_lens_ptr,
    # Q strides
    stride_qb, stride_qh, stride_qd,
    # K cache strides
    stride_kb, stride_kn, stride_kh, stride_kd,
    # V cache strides
    stride_vb, stride_vn, stride_vh, stride_vd,
    # block_table strides
    stride_btb, stride_btm,
    # partial_max strides (B, H_q, SEGMENTS)
    stride_pmb, stride_pmh, stride_pms,
    # partial_lse strides
    stride_plb, stride_plh, stride_pls,
    # partial_out strides (B, H_q, SEGMENTS, d)
    stride_pob, stride_poh, stride_pos, stride_pod,
    scale,
    BLOCK_SIZE:      tl.constexpr,
    HEAD_DIM:        tl.constexpr,
    GQA_GROUP_SIZE:  tl.constexpr,
    PARTITION_SIZE:  tl.constexpr,   # tokens per segment; multiple of BLOCK_SIZE
    IS_FP32:         tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_kv = tl.program_id(axis=1)
    pid_s = tl.program_id(axis=2)

    ctx_len = tl.load(context_lens_ptr + pid_b)

    q_head_start = pid_kv * GQA_GROUP_SIZE
    offs_g = tl.arange(0, GQA_GROUP_SIZE)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = (Q_ptr + pid_b * stride_qb
              + (q_head_start + offs_g)[:, None] * stride_qh
              + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs)
    q_scaled = (q.to(tl.float32) * scale).to(q.dtype)

    m_i = tl.full((GQA_GROUP_SIZE,), value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros((GQA_GROUP_SIZE,), dtype=tl.float32)
    acc = tl.zeros((GQA_GROUP_SIZE, HEAD_DIM), dtype=tl.float32)

    # Segment bounds in blocks.
    BLOCKS_PER_SEG: tl.constexpr = PARTITION_SIZE // BLOCK_SIZE
    seg_block_start = pid_s * BLOCKS_PER_SEG
    num_blocks_total = tl.cdiv(ctx_len, BLOCK_SIZE)
    seg_block_end = tl.minimum(seg_block_start + BLOCKS_PER_SEG,
                               num_blocks_total)

    offs_n = tl.arange(0, BLOCK_SIZE)

    for logical_blk in range(seg_block_start, seg_block_end):
        bt_ptr = block_table_ptr + pid_b * stride_btb + logical_blk * stride_btm
        phys_blk = tl.load(bt_ptr).to(tl.int64)

        token_idx = logical_blk * BLOCK_SIZE + offs_n
        mask_n = token_idx < ctx_len

        k_base = K_cache_ptr + phys_blk * stride_kb + pid_kv * stride_kh
        k_ptrs = (k_base + offs_n[:, None] * stride_kn
                  + offs_d[None, :] * stride_kd)
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

        v_base = V_cache_ptr + phys_blk * stride_vb + pid_kv * stride_vh
        v_ptrs = (v_base + offs_n[:, None] * stride_vn
                  + offs_d[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        if GQA_GROUP_SIZE >= 4 and BLOCK_SIZE >= 16:
            if IS_FP32:
                scores = tl.dot(q_scaled, tl.trans(k), input_precision="ieee")
            else:
                scores = tl.dot(q_scaled, tl.trans(k)).to(tl.float32)
        else:
            q_f = q_scaled.to(tl.float32)
            k_f = k.to(tl.float32)
            scores = tl.sum(q_f[:, None, :] * k_f[None, :, :], axis=2)

        scores = tl.where(mask_n[None, :], scores, -float("inf"))

        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_ij = tl.sum(p, axis=1)
        l_new = alpha * l_i + l_ij

        if GQA_GROUP_SIZE >= 4 and BLOCK_SIZE >= 16:
            if IS_FP32:
                acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
            else:
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
        else:
            v_f = v.to(tl.float32)
            acc = acc * alpha[:, None] + tl.sum(
                p[:, :, None] * v_f[None, :, :], axis=1
            )

        m_i = m_new
        l_i = l_new

    # Write partial (UNNORMALIZED) state.  Always write — invalid segments
    # emit m=-inf, l=0, acc=0, which the reduce kernel cancels via
    # exp(-inf - M_global) = 0.
    pm_ptrs = (partial_max_ptr
               + pid_b * stride_pmb
               + (q_head_start + offs_g) * stride_pmh
               + pid_s * stride_pms)
    pl_ptrs = (partial_lse_ptr
               + pid_b * stride_plb
               + (q_head_start + offs_g) * stride_plh
               + pid_s * stride_pls)
    po_ptrs = (partial_out_ptr
               + pid_b * stride_pob
               + (q_head_start + offs_g)[:, None] * stride_poh
               + pid_s * stride_pos
               + offs_d[None, :] * stride_pod)

    tl.store(pm_ptrs, m_i)
    tl.store(pl_ptrs, l_i)
    tl.store(po_ptrs, acc)


# =============================================================================
# (b2) Split-K reduce kernel.  grid = (B, H_q).
#
# Online-softmax recombination across SEGMENTS using the standard
# alpha = exp(m_s - m_global) rescale. Loads the full (SEGMENTS,) axis
# per program — SEGMENTS is constexpr so the load is a fixed-shape tile.
# =============================================================================

@triton.jit
def paged_attention_reduce_kernel(
    Out_ptr,               # (B, H_q, d) final, native dtype
    partial_max_ptr,       # (B, H_q, SEGMENTS) fp32
    partial_lse_ptr,       # same
    partial_out_ptr,       # (B, H_q, SEGMENTS, d) fp32
    # Out strides
    stride_ob, stride_oh, stride_od,
    # partial_max strides
    stride_pmb, stride_pmh, stride_pms,
    # partial_lse strides
    stride_plb, stride_plh, stride_pls,
    # partial_out strides
    stride_pob, stride_poh, stride_pos, stride_pod,
    HEAD_DIM:    tl.constexpr,
    SEGMENTS:    tl.constexpr,    # actual number of segments
    SEGMENTS_P2: tl.constexpr,    # next power of 2; Triton arange requirement
):
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    offs_s = tl.arange(0, SEGMENTS_P2)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_s = offs_s < SEGMENTS                     # (SEG_P2,)

    base_max = pid_b * stride_pmb + pid_h * stride_pmh
    base_lse = pid_b * stride_plb + pid_h * stride_plh
    base_out = pid_b * stride_pob + pid_h * stride_poh

    # Padded lanes load -inf / 0 / 0 and cancel naturally in the recombine
    # (exp(-inf - m_global) = 0).
    m_s = tl.load(partial_max_ptr + base_max + offs_s * stride_pms,
                  mask=mask_s, other=-float("inf"))
    l_s = tl.load(partial_lse_ptr + base_lse + offs_s * stride_pls,
                  mask=mask_s, other=0.0)
    acc_s = tl.load(
        partial_out_ptr + base_out
        + offs_s[:, None] * stride_pos
        + offs_d[None, :] * stride_pod,
        mask=mask_s[:, None], other=0.0,
    )

    m_global = tl.max(m_s, axis=0)
    alpha = tl.exp(m_s - m_global)
    l_global = tl.sum(alpha * l_s, axis=0)
    acc_global = tl.sum(alpha[:, None] * acc_s, axis=0)

    out = acc_global / l_global

    o_ptrs = (Out_ptr + pid_b * stride_ob + pid_h * stride_oh
              + offs_d * stride_od)
    tl.store(o_ptrs, out.to(Out_ptr.dtype.element_ty))


# =============================================================================
# Python wrapper — dispatches single-pass or split-k.
# =============================================================================

# L4 has 58 SMs. Tune this if you move to a different GPU.
_DEFAULT_SM_COUNT = 58


def _next_pow2(n: int) -> int:
    """Smallest power of 2 >= max(1, n). Used to satisfy Triton's tl.arange
    constraint that the tile length be a power of 2."""
    return 1 << (max(1, n) - 1).bit_length()


def triton_paged_attention_decode(
    Q: torch.Tensor,
    K_cache: torch.Tensor,
    V_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float | None = None,
    use_split_k: bool | None = None,
    partition_size: int = 512,
) -> torch.Tensor:
    """Paged attention forward (decode, MHA or GQA).

    Single-pass: grid = (B, H_kv), each program walks the full sequence.

    Split-k (lesson 12, vLLM v2-style): grid = (B, H_kv, SEGMENTS), each
    program walks PARTITION_SIZE tokens, and a second `reduce` kernel
    over (B, H_q) recombines the per-segment online-softmax states.
    Useful when the base grid (B * H_kv) under-fills the GPU (e.g. MQA
    B=16 H_kv=1 → 16 programs on L4's 58 SMs).

    Args:
        Q: (B, H_q, d) — last axis contiguous.
        K_cache, V_cache: (num_blocks, block_size, H_kv, d).
            H_q must be divisible by H_kv. GQA_GROUP_SIZE = H_q // H_kv.
        block_table: (B, max_blocks_per_seq) int32.
        context_lens: (B,) int32.
        scale: default 1/sqrt(d).
        use_split_k:
            True  → always use split-k path (diagnostic).
            False → always single-pass.
            None  → auto: split-k when (B*H_kv < 0.5*SM_COUNT) and
                    segments >= 4.
        partition_size: tokens per segment for split-k. Must divide
            block_size evenly. Default 512 (vLLM v2 default).

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
    assert partition_size % block_size == 0, (
        f"partition_size={partition_size} must be a multiple of block_size={block_size}"
    )

    if block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)
    if context_lens.dtype != torch.int32:
        context_lens = context_lens.to(torch.int32)

    out = torch.empty_like(Q)
    if scale is None:
        scale = 1.0 / (d ** 0.5)

    is_fp32 = (Q.dtype == torch.float32)

    # ---- Decide path ---------------------------------------------------
    max_ctx = int(context_lens.max().item())
    segments = (max_ctx + partition_size - 1) // partition_size
    single_pass_programs = B * H_kv

    if use_split_k is None:
        # Heuristic tuned empirically on L4 (lesson 12 speed bench):
        #
        #   - Only split when the base grid leaves >= half the SMs idle
        #     (`B*H_kv < 0.5 * SM_COUNT`). If SMs are mostly busy, SK's
        #     extra parallelism doesn't amortize the reduce-kernel launch.
        #   - Require >= 4 segments. With only 2 segments the per-segment
        #     work is too small to pay for the second kernel.
        #
        # Measured on L4 / sm_89:
        #   MQA  (B=16, H_kv=1, ctx=4k)  SP 0.33 ms → SK 0.20 ms  (-39%)
        #   LLaMA-70B (B=4, H_kv=8)       SP 0.17 ms < SK 0.20 ms  (SP better)
        # so the heuristic correctly picks SK for MQA and SP for 70B.
        use_split_k = (
            single_pass_programs < int(_DEFAULT_SM_COUNT * 0.5)
            and segments >= 4
        )

    # A 1-segment "split-k" is just single-pass + launch overhead; force
    # the caller's `True` back to single-pass in that degenerate case.
    # (Callers still passing `use_split_k=True` for diagnostic runs will
    # see segments >= 1; >= 2 is where SK is even meaningful.)
    if use_split_k and segments < 2:
        use_split_k = False

    # ---- Single-pass path ---------------------------------------------
    if not use_split_k:
        grid = (B, H_kv)
        paged_attention_decode_kernel[grid](
            Q, K_cache, V_cache, out,
            block_table, context_lens,
            Q.stride(0),          Q.stride(1),          Q.stride(2),
            K_cache.stride(0),    K_cache.stride(1),    K_cache.stride(2),    K_cache.stride(3),
            V_cache.stride(0),    V_cache.stride(1),    V_cache.stride(2),    V_cache.stride(3),
            block_table.stride(0), block_table.stride(1),
            out.stride(0),        out.stride(1),        out.stride(2),
            scale,
            BLOCK_SIZE=block_size,
            HEAD_DIM=d,
            GQA_GROUP_SIZE=gqa_group_size,
            IS_FP32=is_fp32,
        )
        return out

    # ---- Split-K path --------------------------------------------------
    # Scratch buffers (fp32 for precision in the reduce step).
    partial_max = torch.full(
        (B, H_q, segments), -float("inf"),
        dtype=torch.float32, device=Q.device,
    )
    partial_lse = torch.zeros(
        (B, H_q, segments),
        dtype=torch.float32, device=Q.device,
    )
    partial_out = torch.zeros(
        (B, H_q, segments, d),
        dtype=torch.float32, device=Q.device,
    )

    grid_fwd = (B, H_kv, segments)
    paged_attention_split_kernel[grid_fwd](
        Q, K_cache, V_cache,
        partial_max, partial_lse, partial_out,
        block_table, context_lens,
        Q.stride(0),           Q.stride(1),           Q.stride(2),
        K_cache.stride(0),     K_cache.stride(1),     K_cache.stride(2),     K_cache.stride(3),
        V_cache.stride(0),     V_cache.stride(1),     V_cache.stride(2),     V_cache.stride(3),
        block_table.stride(0), block_table.stride(1),
        partial_max.stride(0), partial_max.stride(1), partial_max.stride(2),
        partial_lse.stride(0), partial_lse.stride(1), partial_lse.stride(2),
        partial_out.stride(0), partial_out.stride(1), partial_out.stride(2), partial_out.stride(3),
        scale,
        BLOCK_SIZE=block_size,
        HEAD_DIM=d,
        GQA_GROUP_SIZE=gqa_group_size,
        PARTITION_SIZE=partition_size,
        IS_FP32=is_fp32,
    )

    grid_red = (B, H_q)
    segments_p2 = _next_pow2(segments)
    paged_attention_reduce_kernel[grid_red](
        out, partial_max, partial_lse, partial_out,
        out.stride(0),         out.stride(1),         out.stride(2),
        partial_max.stride(0), partial_max.stride(1), partial_max.stride(2),
        partial_lse.stride(0), partial_lse.stride(1), partial_lse.stride(2),
        partial_out.stride(0), partial_out.stride(1), partial_out.stride(2), partial_out.stride(3),
        HEAD_DIM=d,
        SEGMENTS=segments,
        SEGMENTS_P2=segments_p2,
    )
    return out
