"""
Lesson 13 / Candidate B — vLLM unified attention vs our lesson-12 kernel.

Goal
----
Test the hypothesis that vLLM's dispatch heuristic
(`seq_threshold_3D = 128 // num_kv_heads`, i.e. trigger split-k whenever
`B * H_kv <= 128`) is calibrated for A100/H100 (108-132 SMs) and misfires
on smaller GPUs like L4 (58 SMs), where `B * H_kv` values in [30, 128]
saturate the device on the single-pass path but vLLM still pays the
split-k reduce-kernel overhead.

Three implementations are compared on the same paged layout, same Q/K/V:

  (1) ours          — triton_paged_attention_decode(..., use_split_k=None)
                      Heuristic: B*H_kv < 0.5*num_SMs AND segments >= 4.
  (2) vllm_default  — extracted vllm.unified_attention(...)
                      with seq_threshold_3D = 128 // num_kv_heads.
                      This is what upstream vLLM actually dispatches with.
  (3) vllm_sm_aware — extracted vllm.unified_attention(...)
                      with seq_threshold_3D = (num_SMs//2) // num_kv_heads.
                      vLLM's own kernel, but re-dispatched with our SM-aware
                      threshold. Isolates "the heuristic is wrong" from
                      "our kernel is faster" — if (3) beats (2), the win is
                      purely from smarter dispatch on vLLM's code.

Expected on L4 (58 SMs):
  - Shapes where B*H_kv ∈ [30, 128] AND ctx ≥ 2k:
      vllm_default goes split-k (wastes reduce launch), ours stays single-pass.
      ours ≈ vllm_sm_aware < vllm_default.
  - Shapes where B*H_kv >> 128 (big batch, prefill-like):
      all three pick single-pass → times within noise.
  - Shapes where B*H_kv ≤ 30 AND ctx >> partition (true MQA decode):
      all three pick split-k → times within noise.

Also includes a correctness smoke test (both forced paths of vllm, ours,
and PyTorch ref must all match within fp16 tolerance).

Run:
    python -m triton_kernels.bench.bench_vllm_vs_ours
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import torch
import triton

from triton_kernels.paged_attention import triton_paged_attention_decode
from triton_kernels.paged_attention_ref import pack_kv_paged, paged_attention_ref
from triton_kernels.vllm_extracted import unified_attention


# =============================================================================
# vLLM wrapper adapter.
# =============================================================================
#
# vLLM's unified_attention() uses a flat-token convention with cu_seqlens_q,
# where Q is shape (num_tokens, num_query_heads, head_size) and cu_seqlens_q
# is [0, q_len_0, q_len_0+q_len_1, ...]. For decode, every sequence has
# q_len == 1, so num_tokens == B and cu_seqlens_q == arange(B+1).
#
# The 3D (split-k) path additionally requires three pre-allocated scratch
# tensors sized by (num_tokens, num_query_heads, NUM_SEGMENTS, head_dim_padded).
# We always allocate them (fine to pass unused when the 2D path runs) so the
# seq_threshold_3D knob is the ONLY thing controlling dispatch.


_NUM_SEGMENTS = 16  # matches vLLM/v1/attention/backends/triton_attn.py:50


def _pow2_up(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def vllm_decode_wrapper(
    Q: torch.Tensor,                 # (B, H_q, d)
    K_cache: torch.Tensor,           # (num_blocks, block_size, H_kv, d)
    V_cache: torch.Tensor,
    block_table: torch.Tensor,       # (B, max_blocks_per_seq) int32
    context_lens: torch.Tensor,      # (B,) int32
    scale: float,
    seq_threshold_3D: int,
) -> torch.Tensor:
    """Adapter: (B, H_q, d) decode → vLLM's flat-token unified_attention."""
    B, H_q, d = Q.shape
    num_blocks, block_size, H_kv, _ = K_cache.shape
    head_size_padded = _pow2_up(d)
    device = Q.device

    out = torch.empty_like(Q)

    cu_seqlens_q = torch.arange(B + 1, dtype=torch.int32, device=device)
    seqused_k = context_lens.to(torch.int32)

    # 3D scratch — always allocated, shaped for worst case.
    segm_output = torch.empty(
        (B, H_q, _NUM_SEGMENTS, head_size_padded), dtype=torch.float32, device=device
    )
    segm_max = torch.empty((B, H_q, _NUM_SEGMENTS), dtype=torch.float32, device=device)
    segm_expsum = torch.empty((B, H_q, _NUM_SEGMENTS), dtype=torch.float32, device=device)

    unified_attention(
        q=Q,
        k=K_cache,
        v=V_cache,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,                        # decode
        seqused_k=seqused_k,
        max_seqlen_k=int(seqused_k.max().item()),
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),                  # no SWA
        block_table=block_table.to(torch.int32),
        softcap=0.0,
        q_descale=None,
        k_descale=1.0,
        v_descale=1.0,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=_NUM_SEGMENTS,
        softmax_segm_output=segm_output,
        softmax_segm_max=segm_max,
        softmax_segm_expsum=segm_expsum,
        alibi_slopes=None,
        output_scale=None,
        qq_bias=None,
        sinks=None,
        mm_prefix_range=None,
    )
    return out


def _predict_vllm_path(num_seqs: int, seq_threshold_3D: int, max_seqlen_q: int = 1) -> str:
    """Replicate vLLM's dispatch branch in unified_attention() for logging."""
    if max_seqlen_q > 1:
        return "2D"
    return "3D" if num_seqs <= seq_threshold_3D else "2D"


def _predict_ours_path(B: int, H_kv: int, max_ctx: int, partition_size: int = 512) -> str:
    from triton_kernels.paged_attention import _DEFAULT_SM_COUNT
    segments = (max_ctx + partition_size - 1) // partition_size
    use_sk = (B * H_kv < int(_DEFAULT_SM_COUNT * 0.5)) and segments >= 4
    return "SK" if use_sk else "SP"


# =============================================================================
# Timing.
# =============================================================================

def _time_ms(fn, warmup: int = 10, iters: int = 50) -> float:
    """Median latency in ms over `iters` CUDA-timed reps."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


# =============================================================================
# Correctness smoke test.
# =============================================================================

@dataclass
class SmokeConfig:
    B: int = 2
    H_q: int = 8
    H_kv: int = 4                       # GQA ratio 2
    head_dim: int = 64
    ctx: int = 768
    block_size: int = 16
    dtype: torch.dtype = torch.float16


def smoke_correctness(cfg: SmokeConfig) -> None:
    torch.manual_seed(0)
    device = "cuda"
    B, H_q, H_kv, d, N = cfg.B, cfg.H_q, cfg.H_kv, cfg.head_dim, cfg.ctx

    Q = torch.randn(B, H_q, d, dtype=cfg.dtype, device=device) * 0.1
    K = torch.randn(B, H_kv, N, d, dtype=cfg.dtype, device=device) * 0.1
    V = torch.randn(B, H_kv, N, d, dtype=cfg.dtype, device=device) * 0.1
    context_lens = torch.full((B,), N, dtype=torch.int32, device=device)

    K_cache, V_cache, block_table, _ = pack_kv_paged(K, V, cfg.block_size, context_lens)
    scale = 1.0 / math.sqrt(d)

    out_ref = paged_attention_ref(Q, K_cache, V_cache, block_table, context_lens, scale)
    out_ours_sp = triton_paged_attention_decode(
        Q, K_cache, V_cache, block_table, context_lens, scale=scale, use_split_k=False
    )
    out_ours_sk = triton_paged_attention_decode(
        Q, K_cache, V_cache, block_table, context_lens, scale=scale, use_split_k=True,
        partition_size=min(512, cfg.block_size * ((N + cfg.block_size - 1) // cfg.block_size)),
    )
    # Force vLLM 2D: threshold=0 → B (=2) > 0, picks 2D.
    out_vllm_2d = vllm_decode_wrapper(
        Q, K_cache, V_cache, block_table, context_lens, scale, seq_threshold_3D=0
    )
    # Force vLLM 3D: threshold=huge → B (=2) <= huge, picks 3D.
    out_vllm_3d = vllm_decode_wrapper(
        Q, K_cache, V_cache, block_table, context_lens, scale, seq_threshold_3D=10**6
    )

    atol, rtol = 5e-3, 5e-3  # fp16 tolerance
    checks = [
        ("ours-SP   vs ref", out_ours_sp, out_ref),
        ("ours-SK   vs ref", out_ours_sk, out_ref),
        ("vllm-2D   vs ref", out_vllm_2d, out_ref),
        ("vllm-3D   vs ref", out_vllm_3d, out_ref),
        ("vllm-2D   vs ours-SP", out_vllm_2d, out_ours_sp),
        ("vllm-3D   vs ours-SK", out_vllm_3d, out_ours_sk),
    ]
    all_ok = True
    print(f"\n[smoke] B={B} H_q={H_q} H_kv={H_kv} d={d} ctx={N} dtype={cfg.dtype}")
    for name, a, b in checks:
        err = (a.float() - b.float()).abs()
        max_err = err.max().item()
        mean_err = err.mean().item()
        ok = torch.allclose(a, b, atol=atol, rtol=rtol)
        flag = "OK" if ok else "FAIL"
        print(f"  [{flag}] {name}: max={max_err:.3e}  mean={mean_err:.3e}")
        if not ok:
            all_ok = False
    if not all_ok:
        raise RuntimeError("correctness smoke failed")


# =============================================================================
# Heuristic-dispatch benchmark.
# =============================================================================

@dataclass
class BenchCase:
    name: str
    B: int
    H_q: int
    H_kv: int
    ctx: int
    head_dim: int = 128
    block_size: int = 16
    dtype: torch.dtype = torch.float16


# Default bench matrix. Each row is a realistic (model, batch, ctx) config
# chosen to exercise the heuristic-mismatch region on L4.
BENCH_CASES: list[BenchCase] = [
    # --- shapes where vLLM's heuristic is EXPECTED to be wrong on L4 ---
    # LLaMA-7B MHA, single-stream decode: H_kv=32, B=1 → B*H_kv=32
    # vLLM: 32 <= 128 → 3D. Ours: 32 >= 29 → SP. Lesson 12 measured: SP wins.
    BenchCase("llama7b_mha_B1_ctx1k",  B=1,  H_q=32, H_kv=32, ctx=1024),
    BenchCase("llama7b_mha_B1_ctx4k",  B=1,  H_q=32, H_kv=32, ctx=4096),
    # LLaMA-70B GQA, modest batch: H_kv=8, B=4 → B*H_kv=32 (same story as above).
    BenchCase("llama70b_gqa_B4_ctx2k", B=4,  H_q=64, H_kv=8,  ctx=2048),
    BenchCase("llama70b_gqa_B4_ctx4k", B=4,  H_q=64, H_kv=8,  ctx=4096),
    # --- shapes where vLLM picks SK correctly (small grid, long ctx) ---
    # MQA-style B=8 H_kv=1 → B*H_kv=8, well under either threshold.
    BenchCase("mqa_B8_ctx4k",          B=8,  H_q=32, H_kv=1,  ctx=4096),
    # --- shapes where vLLM picks SP correctly (big batch or short ctx) ---
    # LLaMA-7B B=32: B*H_kv=1024, both pick SP.
    BenchCase("llama7b_mha_B32_ctx1k", B=32, H_q=32, H_kv=32, ctx=1024),
    # Short-ctx, few-seqs: ctx=256, only 1 segment of 512 → ours will not SK,
    # vLLM will waste 16-way split → ours wins via the ctx-length axis.
    BenchCase("llama7b_mha_B1_ctx256", B=1,  H_q=32, H_kv=32, ctx=256),
]


def bench_one(case: BenchCase, verbose: bool = True) -> dict:
    torch.manual_seed(0)
    device = "cuda"
    B, H_q, H_kv, d, N = case.B, case.H_q, case.H_kv, case.head_dim, case.ctx

    Q = torch.randn(B, H_q, d, dtype=case.dtype, device=device) * 0.1
    K = torch.randn(B, H_kv, N, d, dtype=case.dtype, device=device) * 0.1
    V = torch.randn(B, H_kv, N, d, dtype=case.dtype, device=device) * 0.1
    context_lens = torch.full((B,), N, dtype=torch.int32, device=device)

    K_cache, V_cache, block_table, _ = pack_kv_paged(K, V, case.block_size, context_lens)
    scale = 1.0 / math.sqrt(d)

    from triton_kernels.paged_attention import _DEFAULT_SM_COUNT
    num_sms = _DEFAULT_SM_COUNT
    vllm_default_thresh = 128 // H_kv
    vllm_smaware_thresh = (num_sms // 2) // max(H_kv, 1)

    def run_ours():
        return triton_paged_attention_decode(
            Q, K_cache, V_cache, block_table, context_lens, scale=scale, use_split_k=None
        )

    def run_vllm_default():
        return vllm_decode_wrapper(
            Q, K_cache, V_cache, block_table, context_lens, scale,
            seq_threshold_3D=vllm_default_thresh,
        )

    def run_vllm_smaware():
        return vllm_decode_wrapper(
            Q, K_cache, V_cache, block_table, context_lens, scale,
            seq_threshold_3D=vllm_smaware_thresh,
        )

    t_ours = _time_ms(run_ours)
    t_vllm_def = _time_ms(run_vllm_default)
    t_vllm_sma = _time_ms(run_vllm_smaware)

    path_ours = _predict_ours_path(B, H_kv, N, partition_size=512)
    path_vllm_def = _predict_vllm_path(B, vllm_default_thresh)
    path_vllm_sma = _predict_vllm_path(B, vllm_smaware_thresh)

    if verbose:
        print(
            f"  {case.name:30s}  "
            f"B*H_kv={B*H_kv:4d}  ctx={N:5d}  "
            f"ours[{path_ours}] {t_ours:6.3f}ms   "
            f"vllm-def[{path_vllm_def}] {t_vllm_def:6.3f}ms  ({t_vllm_def/t_ours:4.2f}x ours)   "
            f"vllm-sma[{path_vllm_sma}] {t_vllm_sma:6.3f}ms  ({t_vllm_sma/t_ours:4.2f}x ours)"
        )

    return dict(
        case=case.name,
        B=B, H_q=H_q, H_kv=H_kv, ctx=N, num_sms=num_sms,
        vllm_default_thresh=vllm_default_thresh,
        vllm_smaware_thresh=vllm_smaware_thresh,
        path_ours=path_ours,
        path_vllm_default=path_vllm_def,
        path_vllm_smaware=path_vllm_sma,
        t_ours_ms=t_ours,
        t_vllm_default_ms=t_vllm_def,
        t_vllm_smaware_ms=t_vllm_sma,
    )


# =============================================================================
# Dense-sweep mode: B × H_kv × ctx grid at fixed GQA=4.
# =============================================================================
#
# The primary BENCH_CASES matrix gives 7 hand-picked shapes for a readable
# headline. --sweep runs a denser grid so we can plot the heuristic boundary
# as a heatmap of (B, H_kv) → ratio(vllm-default / ours). Everything at
# H_q = H_kv * 4 (standard GQA), head_dim=128, fp16.
#
# At ~50 iters per config, 72 configs × 3 impls ≈ 15s on L4. Cheap.


def sweep_cases(gqa_group: int = 4) -> list[BenchCase]:
    cases = []
    for ctx in (1024, 4096):
        for H_kv in (1, 2, 4, 8, 16, 32):
            H_q = H_kv * gqa_group
            for B in (1, 2, 4, 8, 16, 32):
                cases.append(BenchCase(
                    name=f"sweep_B{B:02d}_Hkv{H_kv:02d}_Hq{H_q:03d}_ctx{ctx}",
                    B=B, H_q=H_q, H_kv=H_kv, ctx=ctx,
                ))
    return cases


def dump_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def dump_json(rows: list[dict], meta: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-smoke", action="store_true", help="skip correctness check")
    ap.add_argument("--only", type=str, default=None,
                    help="only run bench cases whose name matches this substring")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--sweep", action="store_true",
                    help="run dense B × H_kv × ctx grid on top of primary cases")
    ap.add_argument("--gqa-group", type=int, default=4,
                    help="GQA group size for sweep (H_q = H_kv * gqa_group)")
    ap.add_argument("--out-dir", type=str, default="bench_results",
                    help="directory to write CSV + JSON results (default: bench_results/)")
    ap.add_argument("--quiet-sweep", action="store_true",
                    help="suppress per-row logging for sweep (CSV only)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    props = torch.cuda.get_device_properties(0)
    from triton_kernels.paged_attention import _DEFAULT_SM_COUNT
    print(f"GPU: {props.name}  SMs={props.multi_processor_count}  "
          f"sm_{props.major}{props.minor}  triton={triton.__version__}")
    if props.multi_processor_count != _DEFAULT_SM_COUNT:
        print(f"  NOTE: our heuristic hardcodes _DEFAULT_SM_COUNT={_DEFAULT_SM_COUNT} "
              f"but this GPU has {props.multi_processor_count} SMs. Both 'ours' "
              f"decisions and the vllm-sm-aware threshold use the hardcoded value "
              f"for self-consistency. Bench results are still valid on this GPU "
              f"if you treat _DEFAULT_SM_COUNT as 'the number of SMs the dispatch "
              f"logic thinks it has', not 'the actual SM count'.")

    if not args.skip_smoke:
        smoke_correctness(SmokeConfig())

    print("\n[bench] 3-way dispatch comparison  (time is median of %d iters, 10 warmup)"
          % args.iters)
    print("  legend: path = [SP|SK] for ours, [2D|3D] for vllm; "
          "3D/SK means split-k was chosen")
    print("  threshold: vllm-def = 128 // H_kv (upstream); "
          "vllm-sma = (num_SMs//2) // H_kv  (SM-aware)")

    rows: list[dict] = []

    print("\n[primary] 7 canonical shapes:")
    for case in BENCH_CASES:
        if args.only and args.only not in case.name:
            continue
        rows.append({"phase": "primary", **bench_one(case)})

    if args.sweep:
        cases = sweep_cases(gqa_group=args.gqa_group)
        print(f"\n[sweep] dense grid: {len(cases)} shapes "
              f"(B × H_kv × ctx with H_q = H_kv × {args.gqa_group})")
        for case in cases:
            if args.only and args.only not in case.name:
                continue
            rows.append({"phase": "sweep",
                         **bench_one(case, verbose=not args.quiet_sweep)})

    # --- Summary ---
    print("\n[summary] cases where vLLM's 128 threshold misfires vs SM-aware:")
    any_flag = False
    flagged = []
    for r in rows:
        slower_pct = (r["t_vllm_default_ms"] / r["t_ours_ms"] - 1.0) * 100
        vllm_sm_gap = abs(r["t_vllm_smaware_ms"] / r["t_ours_ms"] - 1.0) * 100
        if slower_pct > 5 and vllm_sm_gap < 10:
            any_flag = True
            flagged.append((slower_pct, r))
    flagged.sort(key=lambda x: -x[0])
    for slower_pct, r in flagged[:20]:
        vllm_sm_gap = abs(r["t_vllm_smaware_ms"] / r["t_ours_ms"] - 1.0) * 100
        print(
            f"  - {r['case']}: vllm-default +{slower_pct:.1f}% vs ours   "
            f"(vllm-sma within {vllm_sm_gap:.1f}%)   "
            f"paths: ours={r['path_ours']} vllm-def={r['path_vllm_default']} "
            f"vllm-sma={r['path_vllm_smaware']}"
        )
    if not any_flag:
        print("  (none — heuristic difference did not materialize on these shapes)")
    elif len(flagged) > 20:
        print(f"  ... +{len(flagged) - 20} more flagged cases in CSV")

    # --- Persist ---
    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"l13_candidateB_stage1_{ts}"
    if args.sweep:
        tag += "_sweep"
    meta = {
        "tag": tag,
        "gpu_name": props.name,
        "gpu_sms_actual": props.multi_processor_count,
        "gpu_sm_capability": f"{props.major}{props.minor}",
        "triton_version": triton.__version__,
        "torch_version": torch.__version__,
        "heuristic_sm_assumed": _DEFAULT_SM_COUNT,
        "warmup": args.warmup,
        "iters": args.iters,
        "gqa_group_for_sweep": args.gqa_group if args.sweep else None,
    }
    csv_path = os.path.join(args.out_dir, f"{tag}.csv")
    json_path = os.path.join(args.out_dir, f"{tag}.json")
    dump_csv(rows, csv_path)
    dump_json(rows, meta, json_path)
    print(f"\n[write] {csv_path}  ({len(rows)} rows)")
    print(f"[write] {json_path}")


if __name__ == "__main__":
    main()
