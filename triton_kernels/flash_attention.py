"""
Lesson 08 · Phase 4 · Flash Attention (forward, non-causal) in Triton.

This is the capstone of the lesson. FA v2 was originally written in Triton
by Tri Dao, which is why the Triton tutorial has a polished implementation.
Our version mirrors Lesson 6's flash_attention_v1 but uses Triton's tl.dot
so the two matmuls go through Tensor Cores on sm_80+.

Comparison with Lesson 6:
  CUDA flash_attention_v1 : fp32 FMA, ~50 lines of manual smem tiling
  Triton this file        : fp16/fp32, ~40 lines, tl.dot -> mma instructions

Algorithm (single head, 2-D Q/K/V of shape (N, d)):

  for each Q block of size BLOCK_M:
      m_i  = -inf                           # running max along N
      l_i  = 0                              # running denominator
      acc  = 0                              # running weighted sum of V

      for each K/V block of size BLOCK_N:
          S     = Q @ K^T * scale           # BLOCK_M x BLOCK_N
          m_ij  = rowmax(S)
          m_new = max(m_i, m_ij)
          alpha = exp(m_i - m_new)
          P     = exp(S - m_new)            # BLOCK_M x BLOCK_N
          l_ij  = rowsum(P)
          l_new = alpha * l_i + l_ij
          acc   = acc * alpha + P @ V
          m_i   = m_new
          l_i   = l_new

      out = acc / l_i
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


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["N", "HEAD_DIM"])
@triton.jit
def flash_attention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qm, stride_qk,
    stride_km, stride_kk,
    stride_vm, stride_vk,
    stride_om, stride_ok,
    N,
    scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Forward Flash Attention. Q, K, V, Out are all (N, HEAD_DIM)."""
    pid_m = tl.program_id(axis=0)

    # -- Load Q tile (BLOCK_M x HEAD_DIM) -----------------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < N

    q_ptrs = Q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # -- Running online-softmax state --------------------------------------
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # -- K/V loop ----------------------------------------------------------
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        k_ptrs = K_ptr + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kk
        v_ptrs = V_ptr + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vk
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # S = Q @ K^T * scale
        # k is (BLOCK_N, HEAD_DIM); we want (HEAD_DIM, BLOCK_N) on the RHS.
        s = tl.dot(q, tl.trans(k)) * scale

        # Mask past-end columns so they don't perturb the max.
        s = tl.where(mask_n[None, :], s, -float("inf"))

        # Online softmax merge.
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_ij = tl.sum(p, axis=1)
        l_new = alpha * l_i + l_ij

        # acc = acc * alpha + P @ V   (accumulate into fp32, cast P for TC)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)

        m_i = m_new
        l_i = l_new

    # Final normalization.
    acc = acc / l_i[:, None]

    # Write output.
    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None])


def triton_flash_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    """Flash Attention forward. Q, K, V must have shape (N, HEAD_DIM)."""
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    assert Q.dim() == 2 and K.dim() == 2 and V.dim() == 2
    assert Q.shape == K.shape == V.shape
    assert Q.dtype == K.dtype == V.dtype
    assert Q.dtype in (torch.float16, torch.float32)
    N, head_dim = Q.shape
    # HEAD_DIM must be a Triton-friendly compile-time constant, power-of-two.
    assert head_dim in (32, 64, 128), f"unsupported head_dim {head_dim}"

    out = torch.empty_like(Q)
    scale = 1.0 / (head_dim ** 0.5)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_M"]),)

    flash_attention_fwd_kernel[grid](
        Q, K, V, out,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        out.stride(0), out.stride(1),
        N, scale,
        HEAD_DIM=head_dim,
    )
    return out


def autotuned_best_config_str() -> str:
    cfg = flash_attention_fwd_kernel.best_config
    k = cfg.kwargs
    return (f"BM={k['BLOCK_M']},BN={k['BLOCK_N']},"
            f"nw={cfg.num_warps},ns={cfg.num_stages}")
