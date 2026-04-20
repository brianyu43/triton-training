"""
Lesson 08 · Phase 1 · Reduction bench.

Runs three reductions over the same input and records the numbers:
  - Triton manual sweep  -> pick best (BLOCK_SIZE, num_warps)
  - torch.sum()
  - CUDA v4_shuffle via subprocess to ./bin/reduction (built from Lesson 3)

Outputs two CSVs to the current working directory when --csv is given:
  reduction_triton_sweep.csv  : every (n, BLOCK_SIZE, num_warps, best_ms)
  reduction_3way.csv          : (n, best_triton_ms, best_triton_cfg,
                                 torch_sum_ms, cuda_v4_ms, cuda_v4_gbps)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import torch

# Make the Triton kernels importable no matter where we run from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from triton_kernels.reduction import (  # noqa: E402
    AUTOTUNE_CONFIGS,
    autotuned_best_config_str,
    triton_reduce_sum,
    triton_reduce_sum_autotuned,
)


# --------------------------------------------------------------------------
# Timing helper (CUDA event based).
# --------------------------------------------------------------------------

def time_ms(fn, warmup: int = 10, iters: int = 50) -> float:
    """Return best-of-N CUDA-event time in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    best = float("inf")
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        ms = start.elapsed_time(end)
        if ms < best:
            best = ms
    return best


# --------------------------------------------------------------------------
# Correctness: autotuned Triton vs torch.sum over one medium tensor.
# --------------------------------------------------------------------------

def correctness(n: int = 1 << 22) -> None:
    torch.manual_seed(0)
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    ref = x.sum().item()

    def _check(label: str, ours: float) -> None:
        abs_err = abs(ours - ref)
        rel_err = abs_err / (abs(ref) + 1e-9)
        print(f"[correctness/{label}]   n = {n}   ours = {ours:.4f}"
              f"   ref = {ref:.4f}   abs_err = {abs_err:.2e}"
              f"   rel_err = {rel_err:.2e}")
        # Sum of 2^22 FP32 Gaussians -> partial-sum ordering matters but
        # 1e-4 relative is far beyond real rounding drift.
        assert rel_err < 1e-4, f"[{label}] correctness failed, rel_err = {rel_err}"

    _check("explicit",  triton_reduce_sum(x, block_size=1024, num_warps=4).item())
    _check("autotuned", triton_reduce_sum_autotuned(x).item())
    print(f"[autotune] chose {autotuned_best_config_str()} for n = {n}")


# --------------------------------------------------------------------------
# Manual sweep of Triton configs.
# --------------------------------------------------------------------------

def sweep_triton(x: torch.Tensor) -> list[dict]:
    rows = []
    for cfg in AUTOTUNE_CONFIGS:
        bs = cfg.kwargs["BLOCK_SIZE"]
        nw = cfg.num_warps
        fn = lambda: triton_reduce_sum(x, block_size=bs, num_warps=nw)
        ms = time_ms(fn)
        rows.append({"BLOCK_SIZE": bs, "num_warps": nw, "ms": ms})
    rows.sort(key=lambda r: r["ms"])
    return rows


# --------------------------------------------------------------------------
# CUDA v4_shuffle via subprocess.
# --------------------------------------------------------------------------

def run_cuda_v4(n: int, binary: str = "./bin/reduction") -> Optional[dict]:
    if not Path(binary).exists():
        return None
    cmd = [binary, "--n", str(n), "--version", "v4", "--csv",
           "--iterations", "50", "--warmup", "10"]
    out = subprocess.check_output(cmd, text=True).strip().splitlines()
    # First line is header, second is data.
    if len(out) < 2:
        return None
    header = [h.strip() for h in out[0].split(",")]
    vals = next(csv.reader([out[1]]))  # handles quoted device name correctly
    row = dict(zip(header, vals))
    return {
        "cuda_v4_ms": float(row["best_ms"]),
        "cuda_v4_gbps": float(row["effective_gbps"]),
        "cuda_v4_efficiency_pct": float(row["efficiency_pct"]),
    }


# --------------------------------------------------------------------------
# Main bench across sizes.
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true",
                    help="write CSVs to current dir and print summary only")
    ap.add_argument("--sizes", type=str, default="1048576,4194304,16777216,67108864",
                    help="comma-separated list of N values")
    ap.add_argument("--cuda-binary", type=str, default="./bin/reduction",
                    help="path to compiled ./bin/reduction from Lesson 3")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s]

    device_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"device = {device_name}   cap = sm_{cap[0]}{cap[1]}")

    # correctness first
    correctness()

    sweep_rows: list[dict] = []
    three_way_rows: list[dict] = []

    for n in sizes:
        torch.manual_seed(n)
        x = torch.randn(n, device="cuda", dtype=torch.float32)

        # manual sweep — records every config so we can see the curve
        cfg_rows = sweep_triton(x)
        for r in cfg_rows:
            sweep_rows.append({"n": n, **r})

        best_cfg = cfg_rows[0]
        best_triton_ms = best_cfg["ms"]
        best_triton_cfg_str = f"BS={best_cfg['BLOCK_SIZE']},nw={best_cfg['num_warps']}"

        # Trigger @triton.autotune so it picks a config for this N — lets us
        # compare autotune's pick vs our manual sweep's pick.
        _ = triton_reduce_sum_autotuned(x)
        autotune_pick = autotuned_best_config_str()

        # torch.sum timing
        torch_ms = time_ms(lambda: x.sum())

        # CUDA v4
        cuda = run_cuda_v4(n, binary=args.cuda_binary)

        # Bandwidth numbers
        bytes_read = n * 4.0
        triton_gbps = bytes_read / (best_triton_ms * 1e-3) / 1e9
        torch_gbps = bytes_read / (torch_ms * 1e-3) / 1e9

        row = {
            "n": n,
            "triton_best_ms": best_triton_ms,
            "triton_best_cfg": best_triton_cfg_str,
            "triton_autotune_cfg": autotune_pick,
            "triton_gbps": triton_gbps,
            "torch_sum_ms": torch_ms,
            "torch_gbps": torch_gbps,
        }
        if cuda is not None:
            row.update(cuda)
        three_way_rows.append(row)

        # Human-readable progress.
        msg = (f"n = {n:>10d}   triton best {best_triton_ms:.3f} ms "
               f"[sweep {best_triton_cfg_str} | autotune {autotune_pick}] "
               f"({triton_gbps:.1f} GB/s)   "
               f"torch.sum {torch_ms:.3f} ms ({torch_gbps:.1f} GB/s)")
        if cuda is not None:
            msg += (f"   CUDA v4 {cuda['cuda_v4_ms']:.3f} ms "
                    f"({cuda['cuda_v4_gbps']:.1f} GB/s)")
        else:
            msg += "   CUDA v4 [binary missing]"
        print(msg)

    if args.csv:
        # Sweep CSV
        sweep_csv = "reduction_triton_sweep.csv"
        with open(sweep_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["n", "BLOCK_SIZE", "num_warps", "ms"])
            w.writeheader()
            for r in sweep_rows:
                w.writerow(r)

        # 3-way CSV
        three_way_csv = "reduction_3way.csv"
        # Compose a union of all keys actually seen (cuda block may be missing).
        fieldnames = [
            "n", "triton_best_ms", "triton_best_cfg", "triton_autotune_cfg",
            "triton_gbps",
            "torch_sum_ms", "torch_gbps",
            "cuda_v4_ms", "cuda_v4_gbps", "cuda_v4_efficiency_pct",
        ]
        with open(three_way_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in three_way_rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})

        print(f"[wrote] {sweep_csv}")
        print(f"[wrote] {three_way_csv}")


if __name__ == "__main__":
    main()
