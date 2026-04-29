"""
Lesson 08 · Phase 2 · Row-wise softmax in Triton.

Comparison with Lesson 4 CUDA progression (v1 naive -> v2 fused -> v3 online):

  v1 naive  : 3 kernels (max, exp, sum/div) = 3 HBM trips per row
  v2 fused  : 1 kernel, whole row in smem    = 1 HBM trip (in + out)
  v3 online : 1 kernel, streaming max + sum  = 1 HBM trip, row size unbounded

Our Triton kernel matches v2_fused in spirit: one program handles one row,
the row lives in registers (not smem), a single-pass safe-softmax pattern:

    x_max = tl.max(x)
    y     = tl.exp(x - x_max)
    y     = y / tl.sum(y)

The "one-pass" here means one HBM read for the row + one HBM write for the
output. That is the exact arithmetic pattern that leads to Flash Attention
(Lesson 6 did the online merge across K-tiles in CUDA — here Triton does
a single tile per row).

BLOCK_SIZE must be >= N_COLS, a power of 2. next_power_of_2(N) is chosen
in the host wrapper. For N > 16384-ish we would need a tiled / online
version, which is Phase 2b if we want it.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------
# Raw kernel (explicit num_warps — used by the manual sweep).
# --------------------------------------------------------------------------

@triton.jit
def softmax_kernel(
    out_ptr,
    in_ptr,
    in_row_stride,
    out_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """One program -> one row. BLOCK_SIZE >= n_cols, power of 2."""
    row = tl.program_id(axis=0)
    in_row = in_ptr + row * in_row_stride
    out_row = out_ptr + row * out_row_stride

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    # -inf for OOB so they don't affect max.
    x = tl.load(in_row + offs, mask=mask, other=-float("inf"))

    # Safe softmax: subtract max before exp.
    x_max = tl.max(x, axis=0)
    x_shifted = x - x_max
    # Exp of -inf is 0, so OOB lanes contribute 0 to the sum.
    y = tl.exp(x_shifted)
    y_sum = tl.sum(y, axis=0)
    y = y / y_sum

    tl.store(out_row + offs, y, mask=mask)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def triton_softmax(
    x: torch.Tensor,
    num_warps: int = 4,
) -> torch.Tensor:
    """Explicit-num_warps row-wise softmax. Returns a new tensor."""
    assert x.is_cuda and x.dim() == 2, "input must be 2-D CUDA tensor"
    assert x.dtype == torch.float32, "only fp32 supported in this phase"

    n_rows, n_cols = x.shape
    block_size = _next_pow2(n_cols)
    out = torch.empty_like(x)

    softmax_kernel[(n_rows,)](
        out, x,
        x.stride(0), out.stride(0),
        n_cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out


# --------------------------------------------------------------------------
# Autotuned variant (production API demo).
# We only tune num_warps; BLOCK_SIZE is dictated by N_COLS. The key is
# BLOCK_SIZE so that rows with different widths get independent tuning.
# --------------------------------------------------------------------------

AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=1),
    triton.Config({}, num_warps=2),
    triton.Config({}, num_warps=4),
    triton.Config({}, num_warps=8),
    triton.Config({}, num_warps=16),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["BLOCK_SIZE"])
@triton.jit
def softmax_kernel_autotuned(
    out_ptr,
    in_ptr,
    in_row_stride,
    out_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    in_row = in_ptr + row * in_row_stride
    out_row = out_ptr + row * out_row_stride

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    x = tl.load(in_row + offs, mask=mask, other=-float("inf"))
    x_max = tl.max(x, axis=0)
    y = tl.exp(x - x_max)
    y = y / tl.sum(y, axis=0)
    tl.store(out_row + offs, y, mask=mask)


def triton_softmax_autotuned(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2
    assert x.dtype == torch.float32
    n_rows, n_cols = x.shape
    block_size = _next_pow2(n_cols)
    out = torch.empty_like(x)
    softmax_kernel_autotuned[(n_rows,)](
        out, x,
        x.stride(0), out.stride(0),
        n_cols,
        BLOCK_SIZE=block_size,
    )
    return out


def autotuned_best_config_str() -> str:
    """Return a short 'nw=...' string for the most recently tuned run."""
    cfg = softmax_kernel_autotuned.best_config
    return f"nw={cfg.num_warps}"
