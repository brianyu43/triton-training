"""
Lesson 09 · Phase 2 · MHA Flash Attention correctness smoke test.

Runs triton_flash_attention_mha across a small grid of (B, H, N, d, dtype,
is_causal) combinations, comparing against F.scaled_dot_product_attention
(fp32 reference).

Phase 2 scope: correctness for both non-causal and causal paths.
Benchmark table (speed comparison) arrives in Phase 3.

Usage (on the L4 VM after repo sync):
    python3 triton_kernels/bench/bench_flash_attention_mha.py

Pass if all rel_err < 1e-2 (fp32) or 5e-2 (fp16).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from triton_kernels.flash_attention_mha import (  # noqa: E402
    autotuned_best_config_str,
    triton_flash_attention_mha,
)


def check(B: int, H: int, N: int, d: int, dtype: torch.dtype,
          is_causal: bool = False) -> bool:
    torch.manual_seed(B * 1_000_003 + H * 10_007 + N * 131 + d
                      + (17 if is_causal else 0))
    q = torch.randn(B, H, N, d, device="cuda", dtype=dtype)
    k = torch.randn(B, H, N, d, device="cuda", dtype=dtype)
    v = torch.randn(B, H, N, d, device="cuda", dtype=dtype)

    # Reference: SDPA in fp32 so rounding of the reference is minimal.
    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    vf = v.to(torch.float32)
    ref = F.scaled_dot_product_attention(qf, kf, vf, is_causal=is_causal)

    ours = triton_flash_attention_mha(q, k, v, is_causal=is_causal).to(torch.float32)
    max_err = (ours - ref).abs().max().item()
    rel_err = max_err / (ref.abs().max().item() + 1e-9)

    tol = 1e-2 if dtype == torch.float32 else 5e-2
    ok = rel_err < tol
    dtype_str = str(dtype).split(".")[-1]
    cstr = "causal" if is_causal else "  full"
    print(f"  B={B} H={H:>2} N={N:>4} d={d:>3} {dtype_str:>7} {cstr}  "
          f"max_err={max_err:.2e}  rel_err={rel_err:.2e}  ok={ok}")
    return ok


def main() -> None:
    device = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"device = {device}   cap = sm_{cap[0]}{cap[1]}")

    # Fair fp32 comparison (same convention as lesson 08).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    all_ok = True

    print("\n=== small shapes, non-causal (catch bugs fast) ===")
    for B in (1, 2, 4):
        for H in (1, 4, 8):
            for N in (128, 512, 1024):
                for dtype in (torch.float32, torch.float16):
                    all_ok = check(B, H, N, 64, dtype, is_causal=False) and all_ok

    print("\n=== small shapes, CAUSAL (Phase 2 new) ===")
    # Include N not divisible by BLOCK_M (129, 513) to stress the diagonal
    # straddling tile at the tail.
    for B in (1, 2):
        for H in (1, 4, 8):
            for N in (128, 129, 512, 513, 1024):
                for dtype in (torch.float32, torch.float16):
                    all_ok = check(B, H, N, 64, dtype, is_causal=True) and all_ok

    print("\n=== realistic LLM shapes (fp16) — non-causal + causal ===")
    # LLaMA-7B inference
    all_ok = check(1, 32, 1024, 128, torch.float16, is_causal=False) and all_ok
    all_ok = check(1, 32, 1024, 128, torch.float16, is_causal=True)  and all_ok
    # LLaMA-7B batched
    all_ok = check(2, 32, 2048, 128, torch.float16, is_causal=False) and all_ok
    all_ok = check(2, 32, 2048, 128, torch.float16, is_causal=True)  and all_ok
    # GPT-2 small
    all_ok = check(8, 12, 1024,  64, torch.float16, is_causal=False) and all_ok
    all_ok = check(8, 12, 1024,  64, torch.float16, is_causal=True)  and all_ok

    # Print the autotune config picked for the last shape to confirm autotune
    # fired across the 4-D grid. (Causal and non-causal are separate entries.)
    print(f"\nautotune picked for last shape: {autotuned_best_config_str()}")

    print("\nALL OK" if all_ok else "FAILED")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
