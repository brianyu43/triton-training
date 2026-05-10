"""
Lesson 11 · Phase 3 + Lesson 12 · split-k — Paged attention speed bench.

Apples-to-apples: for each (B, H, H_kv, d, ctx_len) shape, time
  - (A) torch.nn.functional.scaled_dot_product_attention with enable_gqa=True
        (contiguous KV — the optimistic baseline; cuDNN / FA-2 / aten pick)
  - (B) our triton_paged_attention_decode at each block_size in the sweep,
        via the auto-dispatch path (single-pass when grid saturates L4,
        split-k when it doesn't).

With --compare-paths, we additionally run block_size=16 three times:
single-pass (force off), split-k (force on), and auto. The MQA/low-batch
shapes are the ones expected to benefit from split-k.

Output: markdown table + summary to stdout. Run on the L4 spot VM.

Run:
    python3 triton_kernels/bench/bench_paged_attention_speed.py
    python3 triton_kernels/bench/bench_paged_attention_speed.py --compare-paths
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

    # (B) Our paged kernel at each block_size (auto-dispatch path).
    paged_results = {}
    for bs in BLOCK_SIZES:
        # Pack K/V fresh per block_size.
        K_cache, V_cache, block_table, _ = pack_kv_paged(K, V, bs, ctx_t)

        def run_paged():
            return triton_paged_attention_decode(
                Q, K_cache, V_cache, block_table, ctx_t, scale=scale,
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


def run_path_compare(name: str, B: int, H: int, H_kv: int, d: int, ctx: int,
                     dtype: torch.dtype, device: str, warmup: int, iters: int,
                     block_size: int = 16, partition_size: int = 512):
    """Compare single-pass / split-k / auto paths at one block_size.

    Prints three rows per shape. Expected: MQA and low-batch shapes show
    split-k > single-pass; large-grid shapes show split-k ~ single-pass
    (or slightly worse due to extra kernel launch).
    """
    Q, K, V, ctx_t = make_inputs(B, H, H_kv, d, ctx, dtype, device)
    scale = 1.0 / (d ** 0.5)
    K_cache, V_cache, block_table, _ = pack_kv_paged(K, V, block_size, ctx_t)

    Q4 = Q.unsqueeze(2)

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
    except Exception:
        sdpa_ms = float("nan")

    results = {}
    for mode, use_split_k in [
        ("single-pass", False),
        ("split-k",     True),
        ("auto",        None),
    ]:
        def run_paged(use_split_k=use_split_k):
            return triton_paged_attention_decode(
                Q, K_cache, V_cache, block_table, ctx_t, scale=scale,
                use_split_k=use_split_k,
                partition_size=partition_size,
            )

        try:
            ms = timed(run_paged, warmup, iters)
            gap = (ms - sdpa_ms) / sdpa_ms * 100 if sdpa_ms == sdpa_ms else float("nan")
            results[mode] = (ms, gap)
        except Exception as e:
            results[mode] = (float("nan"), float("nan"))
            print(f"  [warn] paged {mode} failed: {type(e).__name__}: {e}")

    return {
        "name": name, "B": B, "H": H, "H_kv": H_kv, "d": d, "ctx": ctx,
        "block_size": block_size, "partition_size": partition_size,
        "sdpa_ms": sdpa_ms,
        "by_path": results,
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


def print_path_compare(compare_results):
    print()
    print("## Path comparison (block_size=16, partition_size=512)")
    print()
    print("| shape | B | H | H_kv | ctx | SDPA ms "
          "| SP ms | SP gap | SK ms | SK gap | auto ms | auto gap | SK vs SP |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in compare_results:
        sp_ms, sp_gap = r["by_path"]["single-pass"]
        sk_ms, sk_gap = r["by_path"]["split-k"]
        au_ms, au_gap = r["by_path"]["auto"]
        sk_vs_sp = (sp_ms - sk_ms) / sp_ms * 100 if sp_ms == sp_ms and sp_ms > 0 else float("nan")
        print(
            f"| {r['name']} | {r['B']} | {r['H']} | {r['H_kv']} | {r['ctx']} "
            f"| {r['sdpa_ms']:.3f} "
            f"| {sp_ms:.3f} | {sp_gap:+.1f}% "
            f"| {sk_ms:.3f} | {sk_gap:+.1f}% "
            f"| {au_ms:.3f} | {au_gap:+.1f}% "
            f"| {sk_vs_sp:+.1f}% |"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--compare-paths", action="store_true",
                   help="After the block_size sweep, run an extra pass at "
                        "block_size=16 forcing each of single-pass / split-k "
                        "/ auto and print the gap vs SDPA for each path.")
    p.add_argument("--skip-sweep", action="store_true",
                   help="Skip the block_size sweep (only run path compare). "
                        "Implies --compare-paths.")
    args = p.parse_args()
    if args.skip_sweep:
        args.compare_paths = True

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
    if not args.skip_sweep:
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

    if args.compare_paths:
        print("\n" + "=" * 88)
        print("Lesson 12 · path comparison")
        compare_results = []
        for shape in SHAPES:
            name, B, H, H_kv, d, ctx = shape
            print(f"\n--- {name}: B={B} H={H} H_kv={H_kv} d={d} ctx={ctx} ---")
            r = run_path_compare(name, B, H, H_kv, d, ctx, dtype, device,
                                 args.warmup, args.iters,
                                 block_size=16, partition_size=512)
            compare_results.append(r)
            sp_ms, sp_gap = r["by_path"]["single-pass"]
            sk_ms, sk_gap = r["by_path"]["split-k"]
            au_ms, au_gap = r["by_path"]["auto"]
            print(f"  SDPA {r['sdpa_ms']:.3f} ms | "
                  f"SP {sp_ms:.3f} ms ({sp_gap:+.1f}%) | "
                  f"SK {sk_ms:.3f} ms ({sk_gap:+.1f}%) | "
                  f"auto {au_ms:.3f} ms ({au_gap:+.1f}%)")
        print_path_compare(compare_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
