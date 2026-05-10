from __future__ import annotations

import torch

from nanotriton.kernels.rmsnorm import rmsnorm_backward, rmsnorm_forward


class TritonRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("TritonRMSNormFunction expects CUDA input")
        y = rmsnorm_forward(x, weight, eps)
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight = ctx.saved_tensors
        grad_x, grad_weight = rmsnorm_backward(grad_out, x, weight, ctx.eps)
        return grad_x, grad_weight, None


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return TritonRMSNormFunction.apply(x, weight, eps)
