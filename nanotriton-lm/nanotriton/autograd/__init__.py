"""Autograd wrappers around Triton kernels."""

from nanotriton.autograd.rmsnorm_fn import TritonRMSNormFunction
from nanotriton.autograd.swiglu_fn import TritonSwiGLUFunction

__all__ = ["TritonRMSNormFunction", "TritonSwiGLUFunction"]
