"""
Lesson 09 · Phase 2 · Multi-Head Flash Attention (forward, ± causal) in Triton.

Extension of lesson 08's flash_attention.py from 2-D (N, d) to 4-D (B, H, N, d),
plus optional causal masking.

Phase 1 delta vs lesson 08:
  - Kernel takes 4 strides per tensor (B, H, N, d) instead of 2 (N, d).
  - Grid is 3-D: (cdiv(N, BLOCK_M), H, B) instead of 1-D.
  - Each program handles ONE (batch, head) pair's Q block.

Phase 2 delta vs Phase 1:
  - IS_CAUSAL: tl.constexpr splits the kernel into two compiled specializations
    (no runtime branch cost inside the hot loop).
  - Loop-skip optimization: when causal, we only iterate K tiles that overlap
    the lower triangle for this Q block — end_n = min(N, (pid_m+1)*BLOCK_M).
    Tiles that are fully above the diagonal are skipped, saving ~50 % of the
    K/V scan on average. This is the main FA-v2 speedup for causal attention.
  - Diagonal-straddling tile: apply offs_m[:, None] >= offs_n[None, :] mask
    so that entries above the diagonal get -inf before softmax.

Algorithm core (online softmax, tile-by-tile K/V scan, tl.dot matmuls) is the
same as lesson 08. Only the iteration range and mask change for causal.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["N", "HEAD_DIM", "IS_CAUSAL"])
@triton.jit
def flash_attention_mha_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    # Strides in (B, H, N, d) order for each tensor.
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_km, stride_kk,
    stride_vb, stride_vh, stride_vm, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    N,
    scale,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Forward MHA Flash Attention. Q/K/V/Out shape = (B, H, N, HEAD_DIM).

    IS_CAUSAL is constexpr, so Triton compiles two specialized kernels — no
    runtime branch inside the hot loop.
    """
    pid_m = tl.program_id(axis=0)   # which BLOCK_M of queries
    pid_h = tl.program_id(axis=1)   # which head
    pid_b = tl.program_id(axis=2)   # which batch element

    # Shift pointer bases to this (batch, head). After these four lines the
    # kernel body looks exactly like lesson 08's 2-D version.
    q_base = Q_ptr   + pid_b * stride_qb + pid_h * stride_qh
    k_base = K_ptr   + pid_b * stride_kb + pid_h * stride_kh
    v_base = V_ptr   + pid_b * stride_vb + pid_h * stride_vh
    o_base = Out_ptr + pid_b * stride_ob + pid_h * stride_oh

    # -- Load Q tile (BLOCK_M x HEAD_DIM) -----------------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < N

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # -- Running online-softmax state --------------------------------------
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # -- K/V loop range ----------------------------------------------------
    # For causal, the last query row in this program is (pid_m+1)*BLOCK_M - 1.
    # K tiles that start past that row are fully above-diagonal → skip.
    # This is the FA-v2 causal speedup: on average ~50 % of tiles drop out.
    if IS_CAUSAL:
        end_n = tl.minimum(N, (pid_m + 1) * BLOCK_M)
    else:
        end_n = N

    for start_n in range(0, end_n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        k_ptrs = k_base + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kk
        v_ptrs = v_base + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vk
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # S = Q @ K^T * scale
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(mask_n[None, :], s, -float("inf"))

        # Causal mask: zero out entries where col index > row index.
        # Only the diagonal-straddling tile actually has any entries masked
        # here — fully-below tiles are already valid, fully-above tiles are
        # skipped by end_n above. We still emit the mask for all tiles in
        # the causal path because the constexpr branch is folded at compile.
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            s = tl.where(causal_mask, s, -float("inf"))

        # Online softmax merge.
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_ij = tl.sum(p, axis=1)
        l_new = alpha * l_i + l_ij

        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)

        m_i = m_new
        l_i = l_new

    # Final normalization.
    # Causal edge case: for the very first query row (offs_m = 0), only the
    # (0, 0) entry is valid. l_i is nonzero there. For masked-out Q rows
    # (offs_m >= N), l_i is 0 — mask_m guards the store.
    acc = acc / l_i[:, None]

    # Write output.
    out_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None])


def triton_flash_attention_mha(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """MHA Flash Attention forward. Q, K, V must have shape (B, H, N, HEAD_DIM).

    Args:
        Q, K, V: (B, H, N, HEAD_DIM) tensors, fp16 or fp32, last-axis contiguous.
        is_causal: apply causal (lower-triangular) mask. Default False.

    Returns:
        Output tensor of the same shape/dtype as Q.
    """
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    assert Q.dim() == 4 and K.dim() == 4 and V.dim() == 4, (
        f"expected (B, H, N, d); got {Q.shape}, {K.shape}, {V.shape}")
    assert Q.shape == K.shape == V.shape, (
        f"shape mismatch: Q={Q.shape} K={K.shape} V={V.shape}")
    assert Q.dtype == K.dtype == V.dtype
    assert Q.dtype in (torch.float16, torch.float32)

    B, H, N, head_dim = Q.shape
    assert head_dim in (32, 64, 128), f"unsupported head_dim {head_dim}"
    assert (Q.stride(-1) == 1 and K.stride(-1) == 1 and V.stride(-1) == 1), (
        "last axis (head_dim) must be contiguous; call .contiguous() first")

    out = torch.empty_like(Q)
    scale = 1.0 / (head_dim ** 0.5)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_M"]), H, B)

    flash_attention_mha_fwd_kernel[grid](
        Q, K, V, out,
        Q.stride(0),   Q.stride(1),   Q.stride(2),   Q.stride(3),
        K.stride(0),   K.stride(1),   K.stride(2),   K.stride(3),
        V.stride(0),   V.stride(1),   V.stride(2),   V.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        N, scale,
        HEAD_DIM=head_dim,
        IS_CAUSAL=is_causal,
    )
    return out


def autotuned_best_config_str() -> str:
    cfg = flash_attention_mha_fwd_kernel.best_config
    k = cfg.kwargs
    return (f"BM={k['BLOCK_M']},BN={k['BLOCK_N']},"
            f"nw={cfg.num_warps},ns={cfg.num_stages}")
