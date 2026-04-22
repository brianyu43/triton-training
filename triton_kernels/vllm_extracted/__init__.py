# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project (upstream kernels)
# SPDX-FileCopyrightText: Copyright 2026 Xavier (cudatraining lesson 13 extraction)

"""Extracted vLLM Triton kernels, pinned for lesson 13 benchmarking.

See `NOTICE.md` in this directory for the upstream SHA, changes made during
extraction, and license attribution. The import surface intentionally
mirrors upstream so this package can be drop-in compared against vLLM main.
"""

from .unified_attention import (  # noqa: F401
    KVQuantMode,
    kernel_unified_attention_2d,
    kernel_unified_attention_3d,
    reduce_segments,
    unified_attention,
)

__all__ = [
    "KVQuantMode",
    "kernel_unified_attention_2d",
    "kernel_unified_attention_3d",
    "reduce_segments",
    "unified_attention",
]
