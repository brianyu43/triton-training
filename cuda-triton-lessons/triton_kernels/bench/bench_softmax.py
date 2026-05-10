"""
Lesson 08 · Phase 2 · Row-wise softmax bench.

4-way per (M, N) shape:
  - Triton manual sweep over num_warps -> pick best
  - Triton autotuned (reports its pick)
  - torch.softmax (dim=-1)
  - CUDA v2_fused from Lesson 4 via subprocess to ./bin/softmax

Outputs:
  softmax_triton_sweep.csv : every (M, N, num_warps, ms)
  softmax_4way.csv         : (M, N, triton_best_ms, triton_best_cfg,
                              triton_autotune_cfg, torch_softmax_ms,
                              cuda_v2_ms, cuda_v2_gbps)
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from triton_kernels.softmax import (  # noqa: E402
    AUTOTUNE_CONFIGS,
    autotuned_best_config_str,
    triton_softmax,
    triton_softmax_autotuned,
)


def time_ms(fn, warmup: int = 10, iters: int = 50) -> float:
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


def correctness() -> None:
    torch.manual_seed(0)
    for shape in [(64, 128), (512, 1024), (256, 4096)]:
        M, N = shape
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        ref = torch.softmax(x, dim=-1)

        for label, fn in [
            ("explicit",  lambda: triton_softmax(x, num_warps=4)),
            ("autotuned", lambda: triton_softmax_autotuned(x)),
        ]:
            ours = fn()
            max_err = (ours - ref).abs().max().item()
            ok = torch.allclose(ours, ref, atol=1e-6, rtol=1e-5)
            print(f"[correctness/{label}]   shape = {shape}   "
                  f"max_abs_err = {max_err:.2e}   allclose = {ok}")
            assert ok, f"[{label}] softmax mismatch on shape {shape}"

    # Report final autotune pick for the last shape.
    print(f"[autotune] chose {autotuned_best_config_str()}")


def sweep_triton(x: torch.Tensor) -> list[dict]:
    rows = []
    for cfg in AUTOTUNE_CONFIGS:
        nw = cfg.num_warps
        fn = lambda nw=nw: triton_softmax(x, num_warps=nw)
        ms = time_ms(fn)
        rows.append({"num_warps": nw, "ms": ms})
    rows.sort(key=lambda r: r["ms"])
    return rows


def run_cuda_v2(m: int, n: int, binary: str = "./bin/softmax") -> Optional[dict]:
    if not Path(binary).exists():
        return None
    cmd = [binary, "--m", str(m), "--n", str(n), "--version", "v2", "--csv",
           "--iterations", "50", "--warmup", "10"]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip().splitlines()
    except subprocess.CalledProcessError as e:
        print(f"[cuda v2 failed for M={m} N={n}] {e.output}")
        return None
    if len(out) < 2:
        return None
    header = [h.strip() for h in out[0].split(",")]
    vals = next(csv.reader([out[1]]))
    row = dict(zip(header, vals))
    return {
        "cuda_v2_ms": float(row["best_ms"]),
        "cuda_v2_gbps": float(row["effective_gbps"]),
        "cuda_v2_efficiency_pct": float(row["efficiency_pct"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true")
    # Shapes stay within v2's 48KB smem limit (N <= 12288).
    ap.add_argument("--shapes", type=str,
                    default="1024x1024,4096x1024,4096x4096,1024x8192",
                    help="comma-separated MxN list")
    ap.add_argument("--cuda-binary", type=str, default="./bin/softmax")
    args = ap.parse_args()

    device_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"device = {device_name}   cap = sm_{cap[0]}{cap[1]}")

    correctness()

    sweep_rows: list[dict] = []
    four_way_rows: list[dict] = []

    shapes = [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",") if s]

    for (M, N) in shapes:
        torch.manual_seed(M * N)
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)

        cfg_rows = sweep_triton(x)
        for r in cfg_rows:
            sweep_rows.append({"M": M, "N": N, **r})

        best = cfg_rows[0]
        best_triton_ms = best["ms"]
        best_triton_cfg_str = f"nw={best['num_warps']}"

        _ = triton_softmax_autotuned(x)
        autotune_pick = autotuned_best_config_str()

        torch_ms = time_ms(lambda: torch.softmax(x, dim=-1))

        cuda = run_cuda_v2(M, N, binary=args.cuda_binary)

        bytes_moved = M * N * 4 * 2  # read + write
        triton_gbps = bytes_moved / (best_triton_ms * 1e-3) / 1e9
        torch_gbps = bytes_moved / (torch_ms * 1e-3) / 1e9

        row = {
            "M": M, "N": N,
            "triton_best_ms": best_triton_ms,
            "triton_best_cfg": best_triton_cfg_str,
            "triton_autotune_cfg": autotune_pick,
            "triton_gbps": triton_gbps,
            "torch_softmax_ms": torch_ms,
            "torch_gbps": torch_gbps,
        }
        if cuda is not None:
            row.update(cuda)
        four_way_rows.append(row)

        msg = (f"M×N = {M:>4d}×{N:<5d}   "
               f"triton best {best_triton_ms:.3f} ms [sweep {best_triton_cfg_str}"
               f" | autotune {autotune_pick}] ({triton_gbps:.1f} GB/s)   "
               f"torch {torch_ms:.3f} ms ({torch_gbps:.1f} GB/s)")
        if cuda is not None:
            msg += (f"   CUDA v2 {cuda['cuda_v2_ms']:.3f} ms "
                    f"({cuda['cuda_v2_gbps']:.1f} GB/s)")
        else:
            msg += "   CUDA v2 [binary missing]"
        print(msg)

    if args.csv:
        with open("softmax_triton_sweep.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["M", "N", "num_warps", "ms"])
            w.writeheader()
            for r in sweep_rows:
                w.writerow(r)

        fieldnames = [
            "M", "N",
            "triton_best_ms", "triton_best_cfg", "triton_autotune_cfg",
            "triton_gbps",
            "torch_softmax_ms", "torch_gbps",
            "cuda_v2_ms", "cuda_v2_gbps", "cuda_v2_efficiency_pct",
        ]
        with open("softmax_4way.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in four_way_rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})

        print("[wrote] softmax_triton_sweep.csv")
        print("[wrote] softmax_4way.csv")


if __name__ == "__main__":
    main()
