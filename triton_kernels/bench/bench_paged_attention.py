"""
Lesson 11 · Phase 1+2 correctness — Triton paged attention vs references.

Covers MHA (Phase 1) and GQA / MQA (Phase 2: H_q % H_kv == 0 with
H_kv < H_q, LLaMA-3-8B / LLaMA-70B / MQA).

Requires CUDA + Triton. Runs on the GCP L4 spot VM (or any sm_89).

For each test shape:
    1. Generate random contiguous (B, H_q, d) Q and (B, H_kv, N, d) K/V.
    2. Pack K/V into paged layout via pack_kv_paged().
    3. Run:
         a) naive_decode_attention (contiguous, ground truth)
         b) paged_attention_ref    (paged reference, fp32 math)
         c) triton_paged_attention_decode (our Triton kernel)
    4. Verify (c) matches (a) and (b) within fp16/fp32 tolerance.

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

    # Triton kernel (runs in native dtype).
    out_triton = triton_paged_attention_decode(
        Q, K_cache, V_cache, block_table, ctx, scale=scale
    )

    # Tolerances: fp16 is sloppy at softmax edge cases, so allow 1e-2 abs.
    if dtype == torch.float16:
        atol, rtol = 2e-2, 1e-2
    elif dtype == torch.bfloat16:
        atol, rtol = 5e-2, 5e-2
    else:
        atol, rtol = 1e-4, 1e-4

    diff_t_vs_naive  = (out_triton.float() - out_naive.float()).abs().max().item()
    diff_t_vs_ref    = (out_triton.float() - out_ref.float()).abs().max().item()
    diff_ref_vs_naive = (out_ref.float() - out_naive.float()).abs().max().item()

    ok = torch.allclose(out_triton, out_naive, rtol=rtol, atol=atol)
    return {
        "ok": ok,
        "diff_t_vs_naive": diff_t_vs_naive,
        "diff_t_vs_ref": diff_t_vs_ref,
        "diff_ref_vs_naive": diff_ref_vs_naive,
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
    for dtype in (torch.float16, torch.float32):
        print(f"\n--- dtype: {dtype} ---")
        for i, case in enumerate(CASES):
            try:
                r = run_one_case(case, dtype=dtype, device=device)
            except Exception as e:
                print(f"[ERR ] case {i:2d}: {case}   -> {type(e).__name__}: {e}")
                all_pass = False
                continue
            status = "PASS" if r["ok"] else "FAIL"
            ctx_s = r["case"]["context_lens"]
            if len(ctx_s) > 4:
                ctx_s = f"{ctx_s[:2]}..(len={len(ctx_s)})"
            H_kv_display = r["case"].get("H_kv", r["case"]["H"])
            gqa_tag = f"({H_kv_display})" if H_kv_display != r["case"]["H"] else "    "
            print(
                f"[{status}] case {i:2d}: B={r['case']['B']:2d} "
                f"H={r['case']['H']:2d}{gqa_tag} "
                f"d={r['case']['d']:3d} ctx={ctx_s} blk={r['case']['block_size']:3d}  "
                f"triton-naive={r['diff_t_vs_naive']:.2e}  "
                f"triton-ref={r['diff_t_vs_ref']:.2e}"
            )
            if not r["ok"]:
                all_pass = False

    print()
    print("=" * 88)
    if all_pass:
        print("Phase 1 OK — Triton paged attention matches references.")
        return 0
    else:
        print("Phase 1 FAIL — see diffs above.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
