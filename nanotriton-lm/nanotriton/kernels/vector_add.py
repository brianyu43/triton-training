from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements: tl.constexpr, block_size: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {tuple(x.shape)} and {tuple(y.shape)}")
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("vector_add expects CUDA tensors")
    x_contig = x.contiguous()
    y_contig = y.contiguous()
    out = torch.empty_like(x_contig)
    n_elements = out.numel()
    grid = (triton.cdiv(n_elements, block_size),)
    _vector_add_kernel[grid](x_contig, y_contig, out, n_elements, block_size)
    return out.view_as(x)
