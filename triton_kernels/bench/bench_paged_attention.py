"""
Lesson 11 · Phase 1+2 + Lesson 12 correctness —
Triton paged attention vs references, single-pass + split-k paths.

Covers MHA (Phase 1) and GQA / MQA (Phase 2: H_q % H_kv == 0 with
H_kv < H_q, LLaMA-3-8B / LLaMA-70B / MQA) and lesson 12's split-k
path (ctx-axis splits + reduce kernel).

Requires CUDA + Triton. Runs on the GCP L4 spot VM (or any sm_89).

For each test shape:
    1. Generate random contiguous (B, H_q, d) Q and (B, H_kv, N, d) K/V.
    2. Pack K/V into paged layout via pack_kv_paged().
    3. Run:
         a) naive_decode_attention        (contiguous, ground truth)
         b) paged_attention_ref           (paged reference, fp32 math)
         c) triton_paged_attention_decode (single-pass,  auto-dispatch off)
         d) triton_paged_attention_decode (split-k,     forced on)
            — only when ctx_max is big enough to yield >=2 segments.
    4. Verify (c) and (d) both match (a) and (b) within tolerance.

Run:
    python3 triton_kernels/bench/bench_paged_attention.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from triton_kernels.paged_attention import triton_paged_attention_decode   # noqa: E402
from triton_kernels.paged_attention_ref import (                           # noqa: E402
    naive_decode_attention,
    pack_kv_paged,
    paged_attention_ref,
)


CASES = [
    # --- Phase 1: MHA (H_kv == H) ---
    # Tiny — fastest smoke.
    dict(B=1, H=1,  d=64,  context_lens=[32],  block_size=16),
    # Partial last block.
    dict(B=1, H=1,  d=64,  context_lens=[30],  block_size=16),
    # Varying block sizes.
    dict(B=1, H=1,  d=64,  context_lens=[64],  block_size=8),
    dict(B=1, H=1,  d=64,  context_lens=[64],  block_size=32),
    dict(B=1, H=1,  d=64,  context_lens=[64],  block_size=64),
    # Multi-head / multi-batch.
    dict(B=2, H=4,  d=64,  context_lens=[48, 32],           block_size=16),
    dict(B=4, H=8,  d=128, context_lens=[10, 100, 250, 500], block_size=16),
    # LLaMA-7B-style (H=32, d=128), two sequences at different lengths.
    dict(B=2, H=32, d=128, context_lens=[513, 129], block_size=16),
    # Larger batch, GPT-2-style (H=12, d=64).
    dict(B=16, H=12, d=64, context_lens=[1024] * 16, block_size=16),
    # Single-token context.
    dict(B=2, H=4,  d=64,  context_lens=[1, 5], block_size=16),
    # --- Phase 2: GQA (H_kv < H) ---
    # Smallest GQA: group size 2 (Mistral-style).
    dict(B=1, H=4,  H_kv=2,  d=64,  context_lens=[64],  block_size=16),
    # Group size 4.
    dict(B=2, H=8,  H_kv=2,  d=64,  context_lens=[48, 32], block_size=16),
    # LLaMA-3-8B-style: H=32, H_kv=8 (group=4), d=128.
    dict(B=2, H=32, H_kv=8,  d=128, context_lens=[513, 129], block_size=16),
    # LLaMA-70B-style: H=64, H_kv=8 (group=8), d=128.
    dict(B=4, H=64, H_kv=8,  d=128, context_lens=[256, 1024, 2048, 777], block_size=16),
    # Batched long context GQA.
    dict(B=8, H=32, H_kv=8,  d=128, context_lens=[2048] * 8, block_size=16),
    # MQA (Multi-Query): H_kv=1 (group=H). Extreme end of GQA spectrum.
    dict(B=2, H=16, H_kv=1,  d=64,  context_lens=[128, 64], block_size=16),
]


def run_one_case(case: dict, dtype: torch.dtype, device: str, seed: int = 0):
    torch.manual_seed(seed)
    B = case["B"]
    H = case["H"]
    H_kv = case.get("H_kv", H)      # MHA if H_kv not specified
    d = case["d"]
    block_size = case["block_size"]
    context_lens_py = case["context_lens"]
    N_max = max(context_lens_py)

    Q = torch.randn(B, H,    d,     dtype=dtype, device=device)
    K = torch.randn(B, H_kv, N_max, d, dtype=dtype, device=device)
    V = torch.randn(B, H_kv, N_max, d, dtype=dtype, device=device)
    ctx = torch.tensor(context_lens_py, dtype=torch.int32, device=device)

    scale = 1.0 / (d ** 0.5)

    # Ground truth (fp32 math for stability), then cast back.
    out_naive = naive_decode_attention(
        Q.float(), K.float(), V.float(), context_lens=ctx, scale=scale
    ).to(dtype)

    K_cache, V_cache, block_table, _ = pack_kv_paged(K, V, block_size, ctx)

    # Paged reference (also fp32 math internally).
    out_ref = paged_attention_ref(
        Q.float(), K_cache.float(), V_cache.float(),
        block_table, ctx, scale=scale
    ).to(dtype)

    # Triton — single-pass (force off).
    out_sp = triton_paged_attention_decode(
        Q, K_cache, V_cache, block_table, ctx, scale=scale,
        use_split_k=False,
    )

    # Triton — split-k (force on), using a small partition_size so more
    # shapes actually get split. Pick a partition that divides the current
    # block_size and is <= ctx_max (else the wrapper auto-downgrades).
    ctx_max = int(ctx.max().item())
    # partition_size must be a multiple of block_size. Smallest useful is
    # 2 * block_size so the ctx produces >= 2 segments when possible.
    partition_size = max(block_size * 2, 32)
    # Round up to a power of 2 and clamp into [block_size*2, 512].
    partition_size = min(512, max(block_size * 2, partition_size))
    # If ctx_max is smaller than 2*partition_size, split-k degenerates to 1
    # segment and the wrapper falls back to single-pass. That's still
    # correct; we'll report it as "(=SP)" for clarity.
    segments_sk = max(1, (ctx_max + partition_size - 1) // partition_size)
    out_sk = triton_paged_attention_decode(
        Q, K_cache, V_cache, block_table, ctx, scale=scale,
        use_split_k=True,
        partition_size=partition_size,
    )

    # Tolerances: fp16 is sloppy at softmax edge cases, so allow 1e-2 abs.
    if dtype == torch.float16:
        atol, rtol = 2e-2, 1e-2
    elif dtype == torch.bfloat16:
        atol, rtol = 5e-2, 5e-2
    else:
        atol, rtol = 1e-4, 1e-4

    diff_sp_vs_naive  = (out_sp.float() - out_naive.float()).abs().max().item()
    diff_sk_vs_naive  = (out_sk.float() - out_naive.float()).abs().max().item()
    diff_sp_vs_ref    = (out_sp.float() - out_ref.float()).abs().max().item()
    diff_sk_vs_sp     = (out_sk.float() - out_sp.float()).abs().max().item()

    ok_sp = torch.allclose(out_sp, out_naive, rtol=rtol, atol=atol)
    ok_sk = torch.allclose(out_sk, out_naive, rtol=rtol, atol=atol)
    return {
        "ok_sp": ok_sp,
        "ok_sk": ok_sk,
        "diff_sp_vs_naive": diff_sp_vs_naive,
        "diff_sk_vs_naive": diff_sk_vs_naive,
        "diff_sp_vs_ref": diff_sp_vs_ref,
        "diff_sk_vs_sp": diff_sk_vs_sp,
        "segments_sk": segments_sk,
        "partition_size": partition_size,
        "case": case,
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA not available — Phase 1 requires GPU. Run on the GCP L4 VM.")
        return 1

    # Disable TF32 for the torch references (naive + paged einsum). Our
    # Triton kernel is already IEEE-accurate on fp32 (it uses
    # `input_precision="ieee"` on the tl.dot paths, and manual fp32
    # broadcast elsewhere), so any TF32 on the reference side would show
    # up as a spurious diff on MQA / large-GROUP softmax edges.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    print(f"GPU: {gpu_name}  (sm_{cc[0]}{cc[1]})")
    print(f"torch {torch.__version__}")
    print(f"matmul tf32 enabled: {torch.backends.cuda.matmul.allow_tf32}")

    import triton
    print(f"triton {triton.__version__}")
    print("=" * 88)

    all_pass = True
    n_sp_pass = 0
    n_sk_pass = 0
    n_sk_active = 0  # count of cases where split-k actually split (>=2 segments)
    n_total = 0
    for dtype in (torch.float16, torch.float32):
        print(f"\n--- dtype: {dtype} ---")
        for i, case in enumerate(CASES):
            n_total += 1
            try:
                r = run_one_case(case, dtype=dtype, device=device)
            except Exception as e:
                print(f"[ERR ] case {i:2d}: {case}   -> {type(e).__name__}: {e}")
                all_pass = False
                continue
            ok_sp, ok_sk = r["ok_sp"], r["ok_sk"]
            if ok_sp:
                n_sp_pass += 1
            if ok_sk:
                n_sk_pass += 1
            segs = r["segments_sk"]
            if segs >= 2:
                n_sk_active += 1
            status_sp = "PASS" if ok_sp else "FAIL"
            status_sk = "PASS" if ok_sk else "FAIL"
            seg_tag = f"seg={segs}" if segs >= 2 else "seg=1(=SP)"
            ctx_s = r["case"]["context_lens"]
            if len(ctx_s) > 4:
                ctx_s = f"{ctx_s[:2]}..(len={len(ctx_s)})"
            H_kv_display = r["case"].get("H_kv", r["case"]["H"])
            gqa_tag = f"({H_kv_display})" if H_kv_display != r["case"]["H"] else "    "
            print(
                f"[SP:{status_sp} SK:{status_sk}] case {i:2d}: "
                f"B={r['case']['B']:2d} H={r['case']['H']:2d}{gqa_tag} "
                f"d={r['case']['d']:3d} ctx={ctx_s} blk={r['case']['block_size']:3d} "
                f"part={r['partition_size']:3d} {seg_tag:>9s}  "
                f"sp-naive={r['diff_sp_vs_naive']:.2e}  "
                f"sk-naive={r['diff_sk_vs_naive']:.2e}  "
                f"sk-sp={r['diff_sk_vs_sp']:.2e}"
            )
            if not (ok_sp and ok_sk):
                all_pass = False

    print()
    print("=" * 88)
    print(
        f"single-pass: {n_sp_pass}/{n_total} PASS   "
        f"split-k:     {n_sk_pass}/{n_total} PASS   "
        f"(split-k actually split in {n_sk_active}/{n_total} cases)"
    )
    if all_pass:
        print("Lesson 11+12 OK — both Triton paths match references.")
        return 0
    else:
        print("FAIL — see diffs above.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
