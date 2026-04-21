"""
Lesson 11 · Phase 3 — Paged attention speed bench.

Apples-to-apples: for each (B, H, H_kv, d, ctx_len) shape, time
  - (A) torch.nn.functional.scaled_dot_product_attention with enable_gqa=True
        (contiguous KV — the optimistic baseline; cuDNN / FA-2 / aten pick)
  - (B) our triton_paged_attention_decode at each block_size in the sweep

Goal: quantify the overhead of block_table indirection on a realistic
decode workload, and find the block_size sweet spot on L4.

Output: markdown table printed to stdout. Run on the L4 spot VM.

Run:
    python3 triton_kernels/bench/bench_paged_attention_speed.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from triton_kernels.paged_attention import triton_paged_attention_decode   # noqa: E402
from triton_kernels.paged_attention_ref import pack_kv_paged                # noqa: E402


# -----------------------------------------------------------------------------
# Sweep configuration.
# Each entry: (name, B, H, H_kv, d, ctx_len)
# Designed to fit L4 (24GB) with fp16. K/V fp16 size = B*H_kv*ctx*d*2 bytes.
# -----------------------------------------------------------------------------
SHAPES = [
    # Small: batch=1 is the decode baseline (latency-sensitive).
    ("llama7b-B1-ctx1k",   1,  32, 32, 128,  1024),
    ("llama7b-B1-ctx4k",   1,  32, 32, 128,  4096),
    # Medium: throughput-friendly batch.
    ("llama7b-B8-ctx2k",   8,  32, 32, 128,  2048),
    ("llama7b-B32-ctx2k",  32, 32, 32, 128,  2048),
    # Long context.
    ("llama7b-B8-ctx8k",   8,  32, 32, 128,  8192),
    # GQA (LLaMA-3-8B style).
    ("llama38b-B8-ctx2k",  8,  32, 8,  128,  2048),
    ("llama38b-B32-ctx2k", 32, 32, 8,  128,  2048),
    # GQA (LLaMA-70B style).
    ("llama70b-B4-ctx2k",  4,  64, 8,  128,  2048),
    ("llama70b-B8-ctx4k",  8,  64, 8,  128,  4096),
    # MQA extreme.
    ("mqa-B16-ctx4k",      16, 32, 1,  128,  4096),
]

BLOCK_SIZES = [8, 16, 32, 64, 128]


def timed(fn, warmup: int, iters: int) -> float:
    """Mean ms per call using cuda events. Returns ms."""
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bytes_moved(B: int, H: int, H_kv: int, d: int, ctx: int, dtype: torch.dtype) -> float:
    """Approximate bytes moved per decode call for the attention op.

    Q:  B * H * d
    K:  B * H_kv * ctx * d
    V:  B * H_kv * ctx * d
    Out:B * H * d
    """
    itemsize = torch.tensor([], dtype=dtype).element_size()
    return itemsize * (2 * B * H * d + 2 * B * H_kv * ctx * d)


def make_inputs(B: int, H: int, H_kv: int, d: int, ctx: int,
                dtype: torch.dtype, device: str):
    torch.manual_seed(0)
    Q = torch.randn(B, H, d, dtype=dtype, device=device)
    K = torch.randn(B, H_kv, ctx, d, dtype=dtype, device=device)
    V = torch.randn(B, H_kv, ctx, d, dtype=dtype, device=device)
    ctx_t = torch.full((B,), ctx, dtype=torch.int32, device=device)
    return Q, K, V, ctx_t


def run_shape(name: str, B: int, H: int, H_kv: int, d: int, ctx: int,
              dtype: torch.dtype, device: str, warmup: int, iters: int):
    Q, K, V, ctx_t = make_inputs(B, H, H_kv, d, ctx, dtype, device)
    scale = 1.0 / (d ** 0.5)
    bytes_ = bytes_moved(B, H, H_kv, d, ctx, dtype)

    # (A) SDPA contiguous baseline. Use enable_gqa=True so SDPA reads H_kv heads
    # (same data volume as our paged kernel) — fair comparison.
    Q4 = Q.unsqueeze(2)                      # (B, H, 1, d)

    def run_sdpa():
        if H_kv == H:
            return F.scaled_dot_product_attention(
                Q4, K, V, is_causal=False, scale=scale
            )
        return F.scaled_dot_product_attention(
            Q4, K, V, is_causal=False, scale=scale, enable_gqa=True
        )

    try:
        sdpa_ms = timed(run_sdpa, warmup, iters)
        sdpa_gbs = bytes_ / (sdpa_ms * 1e-3) / 1e9
    except Exception as e:
        sdpa_ms = float("nan")
        sdpa_gbs = float("nan")
        print(f"  [warn] SDPA failed on {name}: {type(e).__name__}: {e}")

    # (B) Our paged kernel at each block_size.
    paged_results = {}
    for bs in BLOCK_SIZES:
        # Pack K/V fresh per block_size.
        K_cache, V_cache, block_table, _ = pack_kv_paged(K, V, bs, ctx_t)

        def run_paged():
            return triton_paged_attention_decode(
                Q, K_cache, V_cache, block_table, ctx_t, scale=scale
            )

        try:
            ms = timed(run_paged, warmup, iters)
            gbs = bytes_ / (ms * 1e-3) / 1e9
            speedup = sdpa_ms / ms if ms > 0 else float("nan")
            paged_results[bs] = (ms, gbs, speedup)
        except Exception as e:
            paged_results[bs] = (float("nan"), float("nan"), float("nan"))
            print(f"  [warn] paged bs={bs} failed: {type(e).__name__}: {e}")

    return {
        "name": name, "B": B, "H": H, "H_kv": H_kv, "d": d, "ctx": ctx,
        "sdpa_ms": sdpa_ms, "sdpa_gbs": sdpa_gbs,
        "paged": paged_results,
    }


def print_markdown(results: List[dict]):
    header = (
        "| shape | B | H | H_kv | ctx | SDPA ms | SDPA GB/s | "
        + " | ".join([f"bs={bs} ms / GB/s / x" for bs in BLOCK_SIZES])
        + " |"
    )
    sep = "|" + "|".join(["---"] * (header.count("|") - 1)) + "|"
    print()
    print(header)
    print(sep)
    for r in results:
        cells = [
            r["name"],
            str(r["B"]), str(r["H"]), str(r["H_kv"]), str(r["ctx"]),
            f"{r['sdpa_ms']:.3f}", f"{r['sdpa_gbs']:.1f}",
        ]
        for bs in BLOCK_SIZES:
            ms, gbs, spd = r["paged"][bs]
            cells.append(f"{ms:.3f} / {gbs:.1f} / {spd:.2f}x")
        print("| " + " | ".join(cells) + " |")


def print_summary(results: List[dict]):
    print()
    print("## Summary")
    # Find each shape's best block_size.
    lines = []
    for r in results:
        best_bs = None
        best_ms = float("inf")
        for bs, (ms, _, _) in r["paged"].items():
            if ms == ms and ms < best_ms:    # ms == ms filters NaN
                best_ms = ms
                best_bs = bs
        sdpa = r["sdpa_ms"]
        if best_bs is not None and sdpa == sdpa:
            gap = (best_ms - sdpa) / sdpa * 100
            lines.append(
                f"- {r['name']}: best bs={best_bs} "
                f"({best_ms:.3f} ms, SDPA {sdpa:.3f} ms → paged is "
                f"{gap:+.1f}% vs SDPA)"
            )
    for l in lines:
        print(l)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=200)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available — phase 3 needs GPU.")
        return 1

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"(sm_{torch.cuda.get_device_capability(0)[0]}"
          f"{torch.cuda.get_device_capability(0)[1]})")
    print(f"torch {torch.__version__}")
    import triton
    print(f"triton {triton.__version__}")
    print(f"dtype={args.dtype}  warmup={args.warmup}  iters={args.iters}")
    print(f"block_sizes swept: {BLOCK_SIZES}")

    results = []
    for shape in SHAPES:
        name, B, H, H_kv, d, ctx = shape
        print(f"\n--- {name}: B={B} H={H} H_kv={H_kv} d={d} ctx={ctx} ---")
        r = run_shape(name, B, H, H_kv, d, ctx, dtype, device,
                      args.warmup, args.iters)
        results.append(r)
        # Print one-line summary inline so the terminal shows progress.
        best_bs = None
        best_ms = float("inf")
        for bs, (ms, _, _) in r["paged"].items():
            if ms == ms and ms < best_ms:
                best_ms = ms
                best_bs = bs
        print(f"  SDPA {r['sdpa_ms']:.3f} ms ({r['sdpa_gbs']:.1f} GB/s)  "
              f"best paged bs={best_bs} {best_ms:.3f} ms  "
              f"gap {(best_ms - r['sdpa_ms']) / r['sdpa_ms'] * 100:+.1f}%")

    print_markdown(results)
    print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
