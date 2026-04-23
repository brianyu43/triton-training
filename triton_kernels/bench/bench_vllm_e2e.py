"""
Lesson 13 / Candidate B Stage 2 — vLLM end-to-end decode-heavy bench.

Goal
----
Stage 1/1.5 measured the kernel in isolation on L4. Stage 2 asks: "does the
per-attention-call speedup survive e2e, where attention is only part of the
time budget (alongside MatMuls, softmax, norm, embedding, scheduler, etc.)?"

We run vLLM offline batching with a decode-heavy workload (short prompt,
long generation) on a model that lands its (B * H_kv) values squarely in
the regime where Stage 1 showed upstream dispatch misfires.

Two binaries are compared:
  1. vllm_vanilla   — pip-installed vLLM, unpatched.
  2. vllm_smaware   — same binary with one-line patch at
                      vllm/v1/attention/backends/triton_attn.py:163
                      replacing `seq_threshold_3D = 128 // num_kv_heads`
                      with `seq_threshold_3D = max(num_sms // 2, 1) // num_kv_heads`.

The two are selected by environment variable `VLLM_PATCHED=0|1` which this
script consumes via a side helper that (de)applies the patch against the
installed source tree. We assume the user runs:

    source ~/vllm-venv/bin/activate
    python -m triton_kernels.bench.bench_vllm_e2e --variant vanilla --tag stage2_<ts>
    python -m triton_kernels.bench.bench_vllm_e2e --variant smaware --tag stage2_<ts>

so the patch-state of the installed tree matches the `--variant`.

Output: CSV + JSON to bench_results/ with:
  - total_wall_ms per iteration (warmup excluded)
  - output_tokens generated
  - tokens_per_sec
  - per-variant mean/median/stdev

Workload (decode-heavy, small-batch, long ctx):
  - model    : meta-llama/Llama-3.2-3B-Instruct
               (H_q=24, H_kv=8, d_head=128 → GQA-3, B*H_kv ∈ [8, 32] for B ∈ {1, 2, 4})
  - prompt   : 32-token English instruction template
  - max_tok  : 512 new tokens
  - batch    : 1, 2, 4 sequences (iterates through batches)

This exercises the SP|3D|2D / SK|3D|3D buckets where Stage 1 showed
>20% per-call regression. Expectation: e2e tokens/sec improves by a more
modest but measurable % on small-batch configurations. If the e2e delta is
<1-2 %, the kernel-level gap doesn't translate and the PR lands as a
"correctness-of-heuristic" change rather than a performance win.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

# vLLM is heavy — import lazily so --help is snappy.


DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_PROMPT = (
    "You are a concise assistant. Continue the following text with a "
    "single paragraph of around 200 words, staying on topic. Topic: "
)
DEFAULT_TOPIC = (
    "the history of the printing press and its impact on literacy"
)


@dataclass
class RunConfig:
    model: str
    batch_size: int
    max_new_tokens: int
    iters: int
    warmup: int
    dtype: str
    variant: str            # "vanilla" | "smaware"
    tag: str
    max_model_len: int
    gpu_memory_utilization: float


def build_prompts(batch_size: int) -> list[str]:
    base = DEFAULT_PROMPT + DEFAULT_TOPIC
    # Slightly vary per-slot so vLLM can't dedupe internally.
    return [f"[slot {i}] {base}" for i in range(batch_size)]


def measure_once(llm, prompts: list[str], sampling_params) -> tuple[float, int]:
    """Returns (wall_ms, total_output_tokens)."""
    import torch
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sampling_params, use_tqdm=False)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    tot = sum(len(o.outputs[0].token_ids) for o in outs)
    return (t1 - t0) * 1000.0, tot


def run_variant(cfg: RunConfig) -> dict:
    # Import here so --help avoids torch startup.
    from vllm import LLM, SamplingParams

    prompts = build_prompts(cfg.batch_size)
    sampling = SamplingParams(
        temperature=0.0,                 # greedy for reproducibility
        max_tokens=cfg.max_new_tokens,
        ignore_eos=True,                 # force full decode budget
    )

    print(f"[load] model={cfg.model}  dtype={cfg.dtype}  (variant={cfg.variant})")
    llm = LLM(
        model=cfg.model,
        dtype=cfg.dtype,
        trust_remote_code=True,
        # L4 fits 3B/7B at tight ctx; enforce_eager=False lets CUDA graphs
        # do their thing (matches real deployment). max_model_len is
        # caller-controlled because native context varies by model
        # (TinyLlama = 2048, Qwen/Llama-3 = 4096+).
        enforce_eager=False,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
    )

    print(f"[warmup] {cfg.warmup} iterations of batch={cfg.batch_size}")
    for _ in range(cfg.warmup):
        measure_once(llm, prompts, sampling)

    per_iter_ms: list[float] = []
    per_iter_tok: list[int] = []
    print(f"[measure] {cfg.iters} iterations")
    for i in range(cfg.iters):
        ms, tot = measure_once(llm, prompts, sampling)
        per_iter_ms.append(ms)
        per_iter_tok.append(tot)
        tps = tot / (ms / 1000.0)
        print(f"  iter {i+1:>2d}/{cfg.iters}: {ms:7.1f} ms   {tot:>4d} out_tok   {tps:7.1f} tok/s")

    mean_ms = statistics.mean(per_iter_ms)
    med_ms  = statistics.median(per_iter_ms)
    std_ms  = statistics.stdev(per_iter_ms) if len(per_iter_ms) > 1 else 0.0
    mean_tok = statistics.mean(per_iter_tok)
    tps_med = mean_tok / (med_ms / 1000.0)

    print(f"[summary variant={cfg.variant} batch={cfg.batch_size}] "
          f"median wall = {med_ms:.1f} ms  ±{std_ms:.1f}  "
          f"mean out_tok = {mean_tok:.1f}  median tok/s = {tps_med:.1f}")

    return {
        "variant": cfg.variant,
        "model": cfg.model,
        "batch_size": cfg.batch_size,
        "max_new_tokens": cfg.max_new_tokens,
        "max_model_len": cfg.max_model_len,
        "iters": cfg.iters,
        "warmup": cfg.warmup,
        "dtype": cfg.dtype,
        "wall_ms_mean": mean_ms,
        "wall_ms_median": med_ms,
        "wall_ms_stdev": std_ms,
        "out_tok_mean": mean_tok,
        "tok_per_sec_median": tps_med,
        "per_iter_ms": per_iter_ms,
        "per_iter_tok": per_iter_tok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("vanilla", "smaware"), required=True,
                    help="which vLLM build is currently installed. This script does "
                         "NOT modify the installed tree — use the separate "
                         "scripts/toggle_vllm_patch.sh for that. The flag is "
                         "stamped on the output for bookkeeping.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4],
                    help="batch sizes to iterate")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--max-model-len", type=int, default=2048,
                    help="must be <= model's native context length")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.80,
                    help="vllm KV-cache memory budget as fraction of total VRAM")
    ap.add_argument("--tag", type=str, required=True,
                    help="filename stem for output CSV/JSON in bench_results/")
    ap.add_argument("--out-dir", type=str, default="bench_results")
    args = ap.parse_args()

    results = []
    for B in args.batches:
        cfg = RunConfig(
            model=args.model,
            batch_size=B,
            max_new_tokens=args.max_new_tokens,
            iters=args.iters,
            warmup=args.warmup,
            dtype=args.dtype,
            variant=args.variant,
            tag=args.tag,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        results.append(run_variant(cfg))

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path  = os.path.join(args.out_dir, f"{args.tag}_{args.variant}.csv")
    json_path = os.path.join(args.out_dir, f"{args.tag}_{args.variant}.json")

    # Flat CSV (one row per (batch_size) result; per-iter lists live in JSON only)
    flat = [{k: v for k, v in r.items() if k not in ("per_iter_ms", "per_iter_tok")}
            for r in results]
    if flat:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            w.writeheader()
            for row in flat:
                w.writerow(row)

    meta = {
        "tag": args.tag,
        "variant": args.variant,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(json_path, "w") as f:
        json.dump({"meta": meta, "rows": results}, f, indent=2)

    print(f"\n[write] {csv_path}")
    print(f"[write] {json_path}")


if __name__ == "__main__":
    main()
