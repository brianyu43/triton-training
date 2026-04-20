"""
Lesson 08 · Phase 3 · Block-tiled matmul in Triton.

Comparison with Lesson 5 CUDA progression:
  v1 naive      : 1 thread = 1 output element, terrible reuse
  v2 tiled      : smem tile BM=BN=BK=32, shared reuse
  v3 register   : register-blocked thread tile, more register reuse
  v4 tensor     : WMMA mma.m16n8k16, FP16 inputs + FP32 accumulator

Triton's single `tl.dot(a_tile, b_tile, acc)` replaces all four.

  - Inputs fp32 with allow_tf32=True  -> uses TF32 Tensor Cores on sm_80+
  - Inputs fp16                        -> uses true fp16 Tensor Cores
                                          (mma.m16n8k16, fp32 accumulator)

Grid uses the classic "grouped pid" swizzling trick: instead of iterating
the output tile grid in row-major order (which thrashes L2 when B is large),
group GROUP_SIZE_M rows and walk them before moving to the next row group.
This is the one Triton idiom that has no CUDA analogue you'd write by hand
— you'd get the same L2 hit pattern only by carefully reordering block
indices, which is exactly what this does at the language level.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


AUTOTUNE_CONFIGS = [
    # These configs are a conservative subset of the ones Triton's official
    # matmul tutorial uses for sm_80/sm_89. L4 has 24GB HBM and plenty of
    # SMs so we can afford the larger tiles.
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
                   "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32,
                   "GROUP_SIZE_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
                   "GROUP_SIZE_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 32,
                   "GROUP_SIZE_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
                   "GROUP_SIZE_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32,  "BLOCK_SIZE_K": 32,
                   "GROUP_SIZE_M": 8}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["M", "N", "K"])
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """C = A @ B where A: (M, K), B: (K, N), C: (M, N)."""
    # ------------------------------------------------------------------
    # Grouped program ID → 2D tile coords (pid_m, pid_n) with L2 swizzle.
    # Idea: walk GROUP_SIZE_M rows before moving to the next group,
    # so consecutive programs reuse the same B columns in L2.
    # ------------------------------------------------------------------
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ------------------------------------------------------------------
    # Compute the addresses for the A and B tiles this program consumes.
    # offs_am and offs_bn are masks/indices along the OUTPUT dims; offs_k
    # slides along the contraction dim inside the loop.
    # ------------------------------------------------------------------
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # ------------------------------------------------------------------
    # K-loop: accumulate into an fp32 register tile.
    # ------------------------------------------------------------------
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Masked load so we don't read past K.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # The big moment: compiler chooses the mma instruction based on dtype.
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Cast back to the output dtype (fp32 here — matches c tensor we allocated).
    c = accumulator.to(tl.float32)

    # ------------------------------------------------------------------
    # Write the output tile.
    # ------------------------------------------------------------------
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matmul that dispatches to the autotuned kernel.

    Accepts fp16 or fp32 inputs. Output is always fp32 (the accumulator dtype).
    """
    assert a.is_cuda and b.is_cuda
    assert a.dim() == 2 and b.dim() == 2
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"inner dim mismatch: {K} vs {K2}"
    assert a.dtype == b.dtype, f"dtype mismatch: {a.dtype} vs {b.dtype}"
    assert a.dtype in (torch.float16, torch.float32), "only fp16/fp32 supported"

    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


def autotuned_best_config_str() -> str:
    cfg = matmul_kernel.best_config
    k = cfg.kwargs
    return (f"BM={k['BLOCK_SIZE_M']},BN={k['BLOCK_SIZE_N']},"
            f"BK={k['BLOCK_SIZE_K']},G={k['GROUP_SIZE_M']},"
            f"nw={cfg.num_warps},ns={cfg.num_stages}")
