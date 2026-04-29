from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_forward_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(axis=0)
    offsets = tl.arange(0, block_size)
    mask = offsets < n_cols
    row_start = row * n_cols
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / n_cols
    rstd = tl.rsqrt(variance + eps)
    y = x * rstd * weight
    tl.store(out_ptr + row_start + offsets, y, mask=mask)


@triton.jit
def _rmsnorm_backward_kernel(
    grad_out_ptr,
    x_ptr,
    weight_ptr,
    grad_x_ptr,
    grad_weight_partial_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(axis=0)
    offsets = tl.arange(0, block_size)
    mask = offsets < n_cols
    row_start = row * n_cols
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    grad_out = tl.load(grad_out_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / n_cols
    rstd = tl.rsqrt(variance + eps)
    grad_weight_x = grad_out * weight
    dot = tl.sum(grad_weight_x * x, axis=0)
    grad_x = grad_weight_x * rstd - x * dot * (rstd * rstd * rstd) / n_cols
    grad_weight_partial = grad_out * x * rstd
    tl.store(grad_x_ptr + row_start + offsets, grad_x, mask=mask)
    tl.store(grad_weight_partial_ptr + row_start + offsets, grad_weight_partial, mask=mask)


@triton.jit
def _rmsnorm_dweight_stage1_kernel(
    grad_weight_partial_ptr,
    block_sums_ptr,
    n_rows,
    n_cols: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    row_block = tl.program_id(axis=0)
    col_block = tl.program_id(axis=1)
    row_offsets = row_block * block_m + tl.arange(0, block_m)
    col_offsets = col_block * block_n + tl.arange(0, block_n)
    mask = (row_offsets[:, None] < n_rows) & (col_offsets[None, :] < n_cols)
    values = tl.load(
        grad_weight_partial_ptr + row_offsets[:, None] * n_cols + col_offsets[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    sums = tl.sum(values, axis=0)
    tl.store(
        block_sums_ptr + row_block * n_cols + col_offsets,
        sums,
        mask=col_offsets < n_cols,
    )


@triton.jit
def _rmsnorm_dweight_stage2_kernel(
    block_sums_ptr,
    grad_weight_ptr,
    n_blocks,
    n_cols: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    col_block = tl.program_id(axis=0)
    row_offsets = tl.arange(0, block_m)
    col_offsets = col_block * block_n + tl.arange(0, block_n)
    mask = (row_offsets[:, None] < n_blocks) & (col_offsets[None, :] < n_cols)
    values = tl.load(
        block_sums_ptr + row_offsets[:, None] * n_cols + col_offsets[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    sums = tl.sum(values, axis=0)
    tl.store(grad_weight_ptr + col_offsets, sums, mask=col_offsets < n_cols)


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _num_warps_for_block(block_size: int) -> int:
    if block_size >= 8192:
        return 16
    if block_size >= 2048:
        return 8
    return 4


def _dweight_reduce_block_n(n_cols: int) -> int:
    return min(_next_power_of_2(n_cols), 1024)


def _validate_rmsnorm_inputs(x: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("RMSNorm Triton kernels expect CUDA tensors")
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    if weight.ndim != 1:
        raise ValueError("weight must be a 1-D tensor")
    n_cols = x.shape[-1]
    if weight.numel() != n_cols:
        raise ValueError(f"weight length must match the last x dimension, got {weight.numel()} and {n_cols}")
    block_size = _next_power_of_2(n_cols)
    if block_size > 65536:
        raise ValueError(f"RMSNorm Triton kernels support hidden dimensions up to 65536, got {n_cols}")
    x_contig = x.contiguous()
    weight_contig = weight.contiguous()
    n_rows = x_contig.numel() // n_cols
    return x_contig, weight_contig, n_rows, n_cols, block_size


def _reduce_grad_weight_triton(
    grad_weight_partial: torch.Tensor,
    n_rows: int,
    n_cols: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    row_block_size = 256
    col_block_size = _dweight_reduce_block_n(n_cols)
    row_blocks = triton.cdiv(n_rows, row_block_size)
    col_blocks = triton.cdiv(n_cols, col_block_size)
    block_sums = torch.empty((row_blocks, n_cols), device=grad_weight_partial.device, dtype=torch.float32)
    _rmsnorm_dweight_stage1_kernel[(row_blocks, col_blocks)](
        grad_weight_partial,
        block_sums,
        n_rows,
        n_cols,
        row_block_size,
        col_block_size,
        num_warps=8,
    )

    final_block_m = _next_power_of_2(row_blocks)
    grad_weight = torch.empty((n_cols,), device=grad_weight_partial.device, dtype=dtype)
    _rmsnorm_dweight_stage2_kernel[(col_blocks,)](
        block_sums,
        grad_weight,
        row_blocks,
        n_cols,
        final_block_m,
        col_block_size,
        num_warps=1,
    )
    return grad_weight


def rmsnorm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_contig, weight_contig, n_rows, n_cols, block_size = _validate_rmsnorm_inputs(x, weight)
    out = torch.empty_like(x_contig)
    _rmsnorm_forward_kernel[(n_rows,)](
        x_contig,
        weight_contig,
        out,
        n_cols,
        eps,
        block_size,
        num_warps=_num_warps_for_block(block_size),
    )
    return out.view_as(x)


def rmsnorm_backward(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_contig, weight_contig, n_rows, n_cols, block_size = _validate_rmsnorm_inputs(x, weight)
    if not grad_out.is_cuda:
        raise ValueError("grad_out must be a CUDA tensor")
    if grad_out.shape != x.shape:
        raise ValueError(f"grad_out shape must match x shape, got {tuple(grad_out.shape)} and {tuple(x.shape)}")
    grad_out_contig = grad_out.contiguous()
    grad_x = torch.empty_like(x_contig)
    grad_weight_partial = torch.empty((n_rows, n_cols), device=x.device, dtype=torch.float32)
    _rmsnorm_backward_kernel[(n_rows,)](
        grad_out_contig,
        x_contig,
        weight_contig,
        grad_x,
        grad_weight_partial,
        n_cols,
        eps,
        block_size,
        num_warps=_num_warps_for_block(block_size),
    )
    grad_weight = _reduce_grad_weight_triton(grad_weight_partial, n_rows, n_cols, weight.dtype)
    return grad_x.view_as(x), grad_weight
