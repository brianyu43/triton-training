from __future__ import annotations

import torch

from nanotriton.kernels.swiglu import swiglu_backward, swiglu_forward


class TritonSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if not a.is_cuda or not b.is_cuda:
            raise ValueError("TritonSwiGLUFunction expects CUDA tensors")
        out = swiglu_forward(a, b)
        ctx.save_for_backward(a, b)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a, b = ctx.saved_tensors
        grad_a, grad_b = swiglu_backward(grad_out, a, b)
        return grad_a, grad_b


def triton_swiglu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return TritonSwiGLUFunction.apply(a, b)
