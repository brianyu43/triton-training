"""
Lesson 08 · Phase 4 · Flash Attention bench.

Per N (sequence length), head_dim = 64:
  - Triton FA   (fp32, fp16)
  - torch.nn.functional.scaled_dot_product_attention (fp32, fp16)
  - CUDA flash_attention_v1 (fp32) via ./bin/flash_attention  (Lesson 6)

Output:
  flash_attention_4way.csv

Correctness is checked against torch SDPA (fp32) as reference.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from triton_kernels.flash_attention import (  # noqa: E402
    autotuned_best_config_str,
    triton_flash_attention,
)

HEAD_DIM = 64


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


def sdpa_ref_2d(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """F.scaled_dot_product_attention expects (B, H, N, d). Expand + contract."""
    q4 = q.unsqueeze(0).unsqueeze(0)
    k4 = k.unsqueeze(0).unsqueeze(0)
    v4 = v.unsqueeze(0).unsqueeze(0)
    out4 = F.scaled_dot_product_attention(q4, k4, v4, is_causal=False)
    return out4.squeeze(0).squeeze(0)


def correctness() -> None:
    torch.manual_seed(0)
    for N in (128, 512, 1024):
        for dtype, atol in ((torch.float32, 1e-2), (torch.float16, 5e-2)):
            q = torch.randn(N, HEAD_DIM, device="cuda", dtype=dtype)
            k = torch.randn(N, HEAD_DIM, device="cuda", dtype=dtype)
            v = torch.randn(N, HEAD_DIM, device="cuda", dtype=dtype)

            # Reference in full fp32 for honesty.
            qf = q.to(torch.float32); kf = k.to(torch.float32); vf = v.to(torch.float32)
            ref = sdpa_ref_2d(qf, kf, vf)

            ours = triton_flash_attention(q, k, v).to(torch.float32)
            max_err = (ours - ref).abs().max().item()
            rel_err = max_err / (ref.abs().max().item() + 1e-9)
            ok = rel_err < (1e-2 if dtype == torch.float32 else 5e-2)
            print(f"[correctness/{str(dtype).split('.')[-1]}]   N = {N}   "
                  f"max_abs_err = {max_err:.2e}   rel_err = {rel_err:.2e}   ok = {ok}")
            assert ok, f"FA mismatch at N={N}, dtype={dtype}"


def run_cuda_fa(N: int, binary: str = "./bin/flash_attention") -> Optional[dict]:
    if not Path(binary).exists():
        return None
    # Lesson 6 binary CLI: --n N --version flash --csv --iterations I --warmup W
    cmd = [binary, "--n", str(N), "--version", "flash", "--csv",
           "--iterations", "50", "--warmup", "10"]
    try:
        out = subprocess.check_output(cmd, text=True,
                                      stderr=subprocess.STDOUT).strip().splitlines()
    except subprocess.CalledProcessError as e:
        print(f"[cuda flash N={N} failed] {e.output}")
        return None
    if len(out) < 2:
        return None
    header = [h.strip() for h in out[0].split(",")]
    vals = next(csv.reader([out[1]]))
    row = dict(zip(header, vals))
    # Expected columns include best_ms.
    return {"cuda_flash_ms": float(row.get("best_ms", "nan"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--seq-lens", type=str, default="512,1024,2048,4096,8192")
    ap.add_argument("--cuda-binary", type=str, default="./bin/flash_attention")
    args = ap.parse_args()

    device_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"device = {device_name}   cap = sm_{cap[0]}{cap[1]}   head_dim = {HEAD_DIM}")

    # Enable TF32 for fair fp32 comparison (see Phase 3 discussion).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    correctness()

    seq_lens = [int(v) for v in args.seq_lens.split(",") if v]

    rows = []
    for N in seq_lens:
        torch.manual_seed(N)

        # fp32
        q32 = torch.randn(N, HEAD_DIM, device="cuda", dtype=torch.float32)
        k32 = torch.randn(N, HEAD_DIM, device="cuda", dtype=torch.float32)
        v32 = torch.randn(N, HEAD_DIM, device="cuda", dtype=torch.float32)
        _ = triton_flash_attention(q32, k32, v32); torch.cuda.synchronize()
        triton_fp32_ms = time_ms(lambda: triton_flash_attention(q32, k32, v32))
        triton_fp32_cfg = autotuned_best_config_str()
        torch_fp32_ms = time_ms(lambda: sdpa_ref_2d(q32, k32, v32))

        # fp16
        q16 = q32.to(torch.float16); k16 = k32.to(torch.float16); v16 = v32.to(torch.float16)
        _ = triton_flash_attention(q16, k16, v16); torch.cuda.synchronize()
        triton_fp16_ms = time_ms(lambda: triton_flash_attention(q16, k16, v16))
        triton_fp16_cfg = autotuned_best_config_str()
        torch_fp16_ms = time_ms(lambda: sdpa_ref_2d(q16, k16, v16))

        # CUDA FA v1 (fp32 only per Lesson 6)
        cuda = run_cuda_fa(N, binary=args.cuda_binary)

        # Rough TFLOPS estimate for attention: ~4 * N² * d FLOPs (2 for QK^T, 2 for PV)
        tflops_num = 4.0 * N * N * HEAD_DIM

        def tfl(ms: float) -> float:
            return tflops_num / (ms * 1e-3) / 1e12 if ms > 0 else 0.0

        row = {
            "N": N,
            "triton_fp32_ms": triton_fp32_ms, "triton_fp32_cfg": triton_fp32_cfg,
            "triton_fp32_tflops": tfl(triton_fp32_ms),
            "triton_fp16_ms": triton_fp16_ms, "triton_fp16_cfg": triton_fp16_cfg,
            "triton_fp16_tflops": tfl(triton_fp16_ms),
            "torch_sdpa_fp32_ms": torch_fp32_ms, "torch_sdpa_fp32_tflops": tfl(torch_fp32_ms),
            "torch_sdpa_fp16_ms": torch_fp16_ms, "torch_sdpa_fp16_tflops": tfl(torch_fp16_ms),
        }
        if cuda is not None:
            row["cuda_flash_fp32_ms"] = cuda["cuda_flash_ms"]
            row["cuda_flash_fp32_tflops"] = tfl(cuda["cuda_flash_ms"])
        rows.append(row)

        # Human readable.
        msg = (f"N = {N:>5d}   "
               f"triton fp32 {triton_fp32_ms:6.3f}ms ({tfl(triton_fp32_ms):5.1f}TF)  "
               f"fp16 {triton_fp16_ms:6.3f}ms ({tfl(triton_fp16_ms):5.1f}TF)   "
               f"torch SDPA fp32 {torch_fp32_ms:6.3f}ms  fp16 {torch_fp16_ms:6.3f}ms")
        if cuda is not None:
            msg += f"   CUDA FA fp32 {cuda['cuda_flash_ms']:6.3f}ms"
        print(msg)
        print(f"                    triton autotune: fp32=[{triton_fp32_cfg}]  fp16=[{triton_fp16_cfg}]")

    if args.csv:
        fieldnames = [
            "N",
            "triton_fp32_ms", "triton_fp32_cfg", "triton_fp32_tflops",
            "triton_fp16_ms", "triton_fp16_cfg", "triton_fp16_tflops",
            "torch_sdpa_fp32_ms", "torch_sdpa_fp32_tflops",
            "torch_sdpa_fp16_ms", "torch_sdpa_fp16_tflops",
            "cuda_flash_fp32_ms", "cuda_flash_fp32_tflops",
        ]
        with open("flash_attention_4way.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
        print("[wrote] flash_attention_4way.csv")


if __name__ == "__main__":
    main()
