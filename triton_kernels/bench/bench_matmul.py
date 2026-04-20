"""
Lesson 08 · Phase 3 · Matmul bench — FP32 (TF32 TC) and FP16 (fp16 TC).

Per (M, N, K) shape we run:
  - Triton matmul (autotuned)    — fp32 and fp16 inputs
  - torch.matmul                  — fp32 and fp16
  - CUDA v3_register (fp32)       — via subprocess to ./bin/matmul
  - CUDA v4_tensor (fp16 WMMA TC) — via subprocess

Reports ms, TFLOPS, and autotune's chosen config for Triton.

Output:
  matmul_3way_fp32.csv
  matmul_3way_fp16.csv
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

from triton_kernels.matmul import (  # noqa: E402
    autotuned_best_config_str,
    triton_matmul,
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


def tflops_of(M: int, N: int, K: int, ms: float) -> float:
    """Classical 2*M*N*K FMAs per matmul."""
    return (2.0 * M * N * K) / (ms * 1e-3) / 1e12


def correctness() -> None:
    torch.manual_seed(0)
    for (M, N, K) in [(256, 256, 256), (512, 512, 512), (1024, 768, 512)]:
        for dtype, atol in [(torch.float32, 1e-2), (torch.float16, 5e-2)]:
            a = torch.randn(M, K, device="cuda", dtype=dtype)
            b = torch.randn(K, N, device="cuda", dtype=dtype)
            ours = triton_matmul(a, b)
            ref = (a.to(torch.float32) @ b.to(torch.float32))
            max_err = (ours - ref).abs().max().item()
            # TF32 / fp16 matmul has substantial rounding; we check rel.
            rel_err = max_err / (ref.abs().max().item() + 1e-9)
            ok = rel_err < 1e-2 if dtype == torch.float32 else rel_err < 5e-2
            print(f"[correctness/{str(dtype).split('.')[-1]}]   "
                  f"MNK = ({M},{N},{K})   max_abs_err = {max_err:.2e}   "
                  f"rel_err = {rel_err:.2e}   ok = {ok}")
            assert ok, f"matmul mismatch at {dtype} {(M, N, K)}"


def run_cuda_matmul(M: int, N: int, K: int, version: str,
                    binary: str = "./bin/matmul") -> Optional[dict]:
    if not Path(binary).exists():
        return None
    cmd = [binary, "--m", str(M), "--n", str(N), "--k", str(K),
           "--version", version, "--csv",
           "--iterations", "50", "--warmup", "10"]
    try:
        out = subprocess.check_output(cmd, text=True,
                                      stderr=subprocess.STDOUT).strip().splitlines()
    except subprocess.CalledProcessError as e:
        print(f"[cuda {version} failed] {e.output}")
        return None
    if len(out) < 2:
        return None
    header = [h.strip() for h in out[0].split(",")]
    vals = next(csv.reader([out[1]]))
    row = dict(zip(header, vals))
    # CUDA matmul CSV typically exposes best_ms and tflops already;
    # fall back to computing from best_ms if not.
    best_ms = float(row["best_ms"])
    # CUDA matmul exposes the column as `effective_tflops`.
    if "effective_tflops" in row:
        tflops = float(row["effective_tflops"])
    elif "tflops" in row:
        tflops = float(row["tflops"])
    else:
        tflops = tflops_of(M, N, K, best_ms)
    return {
        f"cuda_{version}_ms":     best_ms,
        f"cuda_{version}_tflops": tflops,
    }


def bench_dtype(dtype: torch.dtype, shapes: list[tuple[int, int, int]],
                cuda_binary: str) -> list[dict]:
    rows = []
    cuda_ver = "v3" if dtype == torch.float32 else "v4"
    for (M, N, K) in shapes:
        torch.manual_seed(M * N + K)
        a = torch.randn(M, K, device="cuda", dtype=dtype)
        b = torch.randn(K, N, device="cuda", dtype=dtype)

        # Warm up once outside timing so autotune settles.
        _ = triton_matmul(a, b)
        torch.cuda.synchronize()
        triton_ms = time_ms(lambda: triton_matmul(a, b))
        triton_cfg = autotuned_best_config_str()
        triton_tflops = tflops_of(M, N, K, triton_ms)

        torch_ms = time_ms(lambda: a @ b)
        torch_tflops = tflops_of(M, N, K, torch_ms)

        cuda = run_cuda_matmul(M, N, K, cuda_ver, binary=cuda_binary)

        row = {
            "M": M, "N": N, "K": K,
            "triton_ms": triton_ms,
            "triton_cfg": triton_cfg,
            "triton_tflops": triton_tflops,
            "torch_ms": torch_ms,
            "torch_tflops": torch_tflops,
        }
        if cuda is not None:
            row.update(cuda)
        rows.append(row)

        dtype_str = str(dtype).split(".")[-1]
        msg = (f"[{dtype_str}]  ({M:>4d},{N:>4d},{K:>4d})   "
               f"triton {triton_ms:.3f} ms ({triton_tflops:5.1f} TF)  [{triton_cfg}]   "
               f"torch {torch_ms:.3f} ms ({torch_tflops:5.1f} TF)")
        if cuda is not None:
            cuda_ms = cuda[f"cuda_{cuda_ver}_ms"]
            cuda_tf = cuda[f"cuda_{cuda_ver}_tflops"]
            msg += f"   CUDA {cuda_ver} {cuda_ms:.3f} ms ({cuda_tf:5.1f} TF)"
        else:
            msg += f"   CUDA {cuda_ver} [binary missing]"
        print(msg)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--shapes", type=str,
                    default="512x512x512,1024x1024x1024,2048x2048x2048,4096x4096x4096")
    ap.add_argument("--cuda-binary", type=str, default="./bin/matmul")
    args = ap.parse_args()

    device_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"device = {device_name}   cap = sm_{cap[0]}{cap[1]}")

    # Fair fp32 comparison: Triton's tl.dot uses TF32 Tensor Cores by default
    # on sm_80+. PyTorch defaults to strict fp32 (no TC). Turn on TF32 for
    # torch so we are comparing the same hardware path on both sides.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"torch.backends.cuda.matmul.allow_tf32 = "
          f"{torch.backends.cuda.matmul.allow_tf32}")

    correctness()

    shapes = [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",") if s]

    fp32_rows = bench_dtype(torch.float32, shapes, args.cuda_binary)
    fp16_rows = bench_dtype(torch.float16, shapes, args.cuda_binary)

    if args.csv:
        for tag, rows, cuda_ver in [("fp32", fp32_rows, "v3"), ("fp16", fp16_rows, "v4")]:
            fieldnames = [
                "M", "N", "K",
                "triton_ms", "triton_cfg", "triton_tflops",
                "torch_ms", "torch_tflops",
                f"cuda_{cuda_ver}_ms", f"cuda_{cuda_ver}_tflops",
            ]
            name = f"matmul_3way_{tag}.csv"
            with open(name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in fieldnames})
            print(f"[wrote] {name}")


if __name__ == "__main__":
    main()
