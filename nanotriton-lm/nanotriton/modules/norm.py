from __future__ import annotations

import torch
import torch.nn as nn

from nanotriton.autograd.rmsnorm_fn import triton_rmsnorm


class TritonRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_rmsnorm(x, self.weight, self.eps)
