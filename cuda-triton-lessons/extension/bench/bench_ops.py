"""Benchmark the custom ops against PyTorch's built-in SDPA.

Measures wall-clock time for three variants at each sequence length:
  (1) torch.ops.mylib.naive_attention   — our 3-kernel naive
  (2) torch.ops.mylib.flash_attention   — our Lesson 06 Flash kernel
  (3) F.scaled_dot_product_attention    — PyTorch's built-in
                                          (on T4 / FP32, cuDNN math path)

Emits CSV to stdout.

Usage:
    python bench/bench_ops.py                  # default sweep
    python bench/bench_ops.py --csv > out.csv
"""

from __future__ import annotations

import argparse
import sys
import time
import torch
import torch.nn.functional as F

import mylib_ext  # noqa: F401


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q4 = q.unsqueeze(0).unsqueeze(0)
    k4 = k.unsqueeze(0).unsqueeze(0)
    v4 = v.unsqueeze(0).unsqueeze(0)
    return F.scaled_dot_product_attention(q4, k4, v4).squeeze(0).squeeze(0)


def time_ms(fn, warmup: int = 10, iters: int = 50) -> float:
    """Return best-of-iters elapsed ms using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(iters):
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        best = min(best, start.elapsed_time(stop))
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", action="store_true", help="emit CSV only")
    parser.add_argument("--sizes", type=str, default="512,1024,2048,4096")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    torch.manual_seed(42)

    sizes = [int(s) for s in args.sizes.split(",")]
    rows = []
    for N in sizes:
        q = torch.randn(N, 64, device=device, dtype=torch.float32)
        k = torch.randn(N, 64, device=device, dtype=torch.float32)
        v = torch.randn(N, 64, device=device, dtype=torch.float32)

        ours_flash = time_ms(lambda: torch.ops.mylib.flash_attention(q, k, v))
        ours_sdpa = time_ms(lambda: sdpa(q, k, v))
        if N <= 2048:  # naive caps at N <= 12288 but CPU-ref infeasible past 2048
            ours_naive = time_ms(lambda: torch.ops.mylib.naive_attention(q, k, v))
        else:
            ours_naive = float("nan")

        rows.append((N, ours_naive, ours_flash, ours_sdpa))

    if args.csv:
        print("n,ours_naive_ms,ours_flash_ms,torch_sdpa_ms")
        for (n, a, b, c) in rows:
            print(f"{n},{a:.6f},{b:.6f},{c:.6f}")
    else:
        print(f"{'N':>6} | {'naive (ms)':>11} | {'flash (ms)':>11} | "
              f"{'sdpa (ms)':>11} | flash vs sdpa")
        print("-" * 64)
        for (n, a, b, c) in rows:
            ratio = c / b if b > 0 else float("nan")
            a_str = f"{a:11.3f}" if a == a else f"{'--':>11}"
            print(f"{n:>6} | {a_str} | {b:11.3f} | {c:11.3f} | {ratio:5.2f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main())
