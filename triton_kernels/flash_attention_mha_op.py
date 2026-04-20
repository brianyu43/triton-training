"""
Lesson 09 · Phase 4 · torch.library.custom_op registration.

Wraps the Phase 1-2 `triton_flash_attention_mha` function as a first-class
PyTorch operator living under the `triton_training` namespace.

After `import triton_kernels.flash_attention_mha_op`, the kernel is callable as:

    torch.ops.triton_training.flash_attention_mha(q, k, v, is_causal)

What this unlocks vs calling the raw Python wrapper:

1. torch.compile graph continuity. Dynamo sees a registered custom op, uses
   the `register_fake` impl for shape inference, and keeps the surrounding
   graph intact — no graph break at our kernel.

2. torch.export / serialization. The op name is part of the exported graph,
   so export artifacts point to `triton_training::flash_attention_mha`.

3. Clean drop-in for downstream packages. Anyone who `import`s this module
   can call via the standard `torch.ops.<namespace>.<op>` path without
   touching Triton-specific Python code.

What this does NOT add:
- backward (autograd). We only implemented forward — training use requires
  a separate backward kernel + register_autograd.
- CPU impl. CUDA-only.

Reference: https://pytorch.org/docs/stable/library.html
"""

from __future__ import annotations

import torch
from torch.library import custom_op

from triton_kernels.flash_attention_mha import triton_flash_attention_mha


# `mutates_args=()` declares pure function (no in-place mutation of inputs).
# `device_types="cuda"` tells the dispatcher this op is only registered for
# CUDA tensors; CPU calls will raise a clear error instead of silently
# falling through.
@custom_op(
    "triton_training::flash_attention_mha",
    mutates_args=(),
    device_types="cuda",
)
def flash_attention_mha_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """Forward MHA Flash Attention. Q, K, V: (B, H, N, HEAD_DIM), fp16/fp32."""
    return triton_flash_attention_mha(q, k, v, is_causal=is_causal)


@flash_attention_mha_op.register_fake
def _flash_attention_mha_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """Fake / meta implementation used by torch.compile for shape inference.

    Runs on FakeTensors (no data) — cannot launch the Triton kernel here.
    Just declare the output shape / dtype / device, and dynamo handles the
    rest during graph capture.
    """
    # Output matches Q exactly: same (B, H, N, d), same dtype, same device.
    return torch.empty_like(q)


__all__ = ["flash_attention_mha_op"]
