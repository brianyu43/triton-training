from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_forward_kernel(a_ptr, b_ptr, out_ptr, n_elements: tl.constexpr, block_size: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-a))
    out = a * sigmoid * b
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _swiglu_backward_kernel(
    grad_out_ptr,
    a_ptr,
    b_ptr,
    grad_a_ptr,
    grad_b_ptr,
    n_elements: tl.constexpr,
    block_size: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-a))
    silu = a * sigmoid
    dsilu = sigmoid * (1.0 + a * (1.0 - sigmoid))
    grad_a = grad_out * b * dsilu
    grad_b = grad_out * silu
    tl.store(grad_a_ptr + offsets, grad_a, mask=mask)
    tl.store(grad_b_ptr + offsets, grad_b, mask=mask)


def _validate_swiglu_inputs(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("SwiGLU Triton kernels expect CUDA tensors")
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape, got {tuple(a.shape)} and {tuple(b.shape)}")
    a_contig = a.contiguous()
    b_contig = b.contiguous()
    return a_contig, b_contig, a_contig.numel()


def swiglu_forward(a: torch.Tensor, b: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    a_contig, b_contig, n_elements = _validate_swiglu_inputs(a, b)
    out = torch.empty_like(a_contig)
    grid = (triton.cdiv(n_elements, block_size),)
    _swiglu_forward_kernel[grid](a_contig, b_contig, out, n_elements, block_size)
    return out.view_as(a)


def swiglu_backward(
    grad_out: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    block_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    a_contig, b_contig, n_elements = _validate_swiglu_inputs(a, b)
    if not grad_out.is_cuda:
        raise ValueError("grad_out must be a CUDA tensor")
    if grad_out.shape != a.shape:
        raise ValueError(f"grad_out shape must match a and b, got {tuple(grad_out.shape)} and {tuple(a.shape)}")
    grad_out_contig = grad_out.contiguous()
    grad_a = torch.empty_like(a_contig)
    grad_b = torch.empty_like(b_contig)
    grid = (triton.cdiv(n_elements, block_size),)
    _swiglu_backward_kernel[grid](
        grad_out_contig,
        a_contig,
        b_contig,
        grad_a,
        grad_b,
        n_elements,
        block_size,
    )
    return grad_a.view_as(a), grad_b.view_as(b)
