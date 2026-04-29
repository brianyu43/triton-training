"""
Lesson 08 · Phase 0 · Triton smoke test.

Goal: verify that Triton JIT compiles and runs on this GPU.

What we exercise:
  - @triton.jit decorator (Python -> PTX through Triton)
  - tl.program_id : block index along grid axis 0 (analogous to CUDA blockIdx.x)
  - tl.arange + mask : tail handling without branching
  - tl.load / tl.store with mask argument
  - Grid computed as a function of meta-parameters (lambda META: ...)

Mental model reminder:
  In CUDA each thread computes one element; in Triton each *program instance*
  handles a tile of BLOCK_SIZE elements, and the compiler figures out the
  per-thread assignment. One program = one tile, not one element.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    x_ptr,          # *float32  input A
    y_ptr,          # *float32  input B
    out_ptr,        # *float32  output
    n_elements,     # int       total length
    BLOCK_SIZE: tl.constexpr,   # compile-time constant, one value per autotune config
):
    """One program instance handles BLOCK_SIZE contiguous elements."""
    pid = tl.program_id(axis=0)                       # which tile am I?
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # vector of indices for this tile
    mask = offsets < n_elements                       # guard the last (partial) tile

    x = tl.load(x_ptr + offsets, mask=mask)           # masked load -> 0 for OOB lanes
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda, "inputs must be CUDA tensors"
    assert x.shape == y.shape, "shape mismatch"
    assert x.is_contiguous() and y.is_contiguous(), "inputs must be contiguous"

    out = torch.empty_like(x)
    n = x.numel()

    # Grid is a lambda so Triton can expand meta-params at autotune time.
    # Here we have only one config, so it is trivially ceil_div(n, BLOCK_SIZE).
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    vector_add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
    return out


def main() -> None:
    torch.manual_seed(0)

    # Device + runtime sanity report.
    device_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"device = {device_name}   cap = sm_{cap[0]}{cap[1]}")
    print(f"torch  = {torch.__version__}   triton = {triton.__version__}")

    # Small + medium + not-multiple-of-BLOCK_SIZE (tests the mask tail).
    sizes = [1024, 1 << 16, 1_000_003]
    for n in sizes:
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        y = torch.randn(n, device="cuda", dtype=torch.float32)

        out_triton = vector_add(x, y)
        out_ref = x + y

        max_abs_err = (out_triton - out_ref).abs().max().item()
        ok = torch.allclose(out_triton, out_ref)
        print(f"n = {n:>10d}   max_abs_err = {max_abs_err:.2e}   allclose = {ok}")
        assert ok, f"mismatch at n={n}"

    print("smoke OK")


if __name__ == "__main__":
    main()
