"""
Lesson 08 · Phase 1 · Triton sum reduction.

Compare with Lesson 3 CUDA progression (v1 atomic -> v4 warp-shuffle):
  one Triton program instance ~= one CUDA thread block.
  `tl.sum(x, axis=0)` replaces the entire warp-shuffle + smem tree by
  letting the compiler pick the right instruction sequence for the given
  BLOCK_SIZE and num_warps.

Two-pass design:
  pass 1 : N elements -> ceil(N / BLOCK_SIZE) partial sums  (Triton)
  pass 2 : partial sums -> scalar                          (torch.sum)

The second pass operates on at most ~N/MIN_BLOCK_SIZE elements, which for
N=2^26 and BLOCK_SIZE=1024 is 65536 — negligible compared to pass 1.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------
# Raw kernel (explicit config — used by the manual sweep in the bench).
# --------------------------------------------------------------------------

@triton.jit
def reduce_sum_kernel(
    x_ptr,
    partial_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """One program -> partial sum of BLOCK_SIZE elements -> partial_ptr[pid]."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    partial = tl.sum(x, axis=0)
    tl.store(partial_ptr + pid, partial)


def triton_reduce_sum(
    x: torch.Tensor,
    block_size: int = 1024,
    num_warps: int = 4,
) -> torch.Tensor:
    """Explicit-config Triton sum reduction. Returns a 0-dim CUDA tensor."""
    assert x.is_cuda and x.is_contiguous(), "input must be contiguous CUDA"
    assert x.dtype == torch.float32, "only fp32 supported in this phase"

    n = x.numel()
    num_programs = triton.cdiv(n, block_size)
    partial = torch.empty(num_programs, device=x.device, dtype=x.dtype)

    grid = (num_programs,)
    reduce_sum_kernel[grid](
        x, partial, n,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return partial.sum()


# --------------------------------------------------------------------------
# Autotuned variant — for the production API demo.
# The autotuner runs each config once on the first call, picks the fastest,
# and caches the decision keyed on `n_elements`.
# --------------------------------------------------------------------------

AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 256},  num_warps=2),
    triton.Config({"BLOCK_SIZE": 512},  num_warps=4),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
]


@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=["n_elements"],
    # Zero the partial buffer before each trial run during autotuning so
    # trials don't contaminate each other.
    reset_to_zero=["partial_ptr"],
)
@triton.jit
def reduce_sum_kernel_autotuned(
    x_ptr,
    partial_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    partial = tl.sum(x, axis=0)
    tl.store(partial_ptr + pid, partial)


def triton_reduce_sum_autotuned(x: torch.Tensor) -> torch.Tensor:
    """Autotuned Triton sum reduction. First call sweeps configs.

    We slice `partial` to exactly `num_programs_for_best_config` before the
    second-pass sum, so even if autotuning left garbage in the tail of the
    buffer (from trial runs with larger BLOCK_SIZEs), we ignore it.
    """
    assert x.is_cuda and x.is_contiguous(), "input must be contiguous CUDA"
    assert x.dtype == torch.float32

    n = x.numel()
    # Preallocate based on the smallest BLOCK_SIZE in our configs so the
    # chosen config never writes past the end.
    min_block = min(cfg.kwargs["BLOCK_SIZE"] for cfg in AUTOTUNE_CONFIGS)
    max_programs = triton.cdiv(n, min_block)
    partial = torch.empty(max_programs, device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    reduce_sum_kernel_autotuned[grid](x, partial, n)

    # Only the prefix written by the *chosen* config is meaningful.
    best_cfg = reduce_sum_kernel_autotuned.best_config
    block = best_cfg.kwargs["BLOCK_SIZE"]
    num_programs = triton.cdiv(n, block)
    return partial[:num_programs].sum()


def autotuned_best_config_str() -> str:
    """Return a short 'BS=...,nw=...' string for the most recently tuned run."""
    cfg = reduce_sum_kernel_autotuned.best_config
    return f"BS={cfg.kwargs['BLOCK_SIZE']},nw={cfg.num_warps}"
