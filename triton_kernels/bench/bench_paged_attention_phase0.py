"""
Lesson 11 · Phase 0 correctness — paged reference vs contiguous attention.

No CUDA required; runs fp32 on CPU for a tight numerical match. Verifies
that pack_kv_paged + paged_attention_ref matches the naive contiguous
decode attention over a spread of shapes (including partial-last-block
context lengths).

Run:
    python3 triton_kernels/bench/bench_paged_attention_phase0.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import torch

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from triton_kernels.paged_attention_ref import (  # noqa: E402
    naive_decode_attention,
    pack_kv_paged,
    paged_attention_ref,
)


# -----------------------------------------------------------------------------
# Test cases — (B, H, d, context_lens, block_size).
# context_lens is a list of per-sequence lengths to exercise variable-length
# batches and partial last blocks.
# -----------------------------------------------------------------------------

CASES = [
    # Smallest — single seq, single head, two full blocks.
    dict(B=1, H=1, d=64,  context_lens=[32], block_size=16),
    # Partial last block (30 tokens in two 16-blocks = 16 full + 14 partial).
    dict(B=1, H=1, d=64,  context_lens=[30], block_size=16),
    # Different block sizes.
    dict(B=1, H=1, d=64,  context_lens=[64], block_size=8),
    dict(B=1, H=1, d=64,  context_lens=[64], block_size=32),
    # Multi-head.
    dict(B=1, H=4, d=64,  context_lens=[48], block_size=16),
    # Multi-batch, variable lengths (classic vLLM decode scenario).
    dict(B=3, H=4, d=64,  context_lens=[16, 31, 48], block_size=16),
    dict(B=4, H=8, d=128, context_lens=[10, 100, 250, 500], block_size=16),
    # Larger head dim (common in LLaMA-7B).
    dict(B=2, H=32, d=128, context_lens=[513, 129], block_size=16),
    # Single token context (edge — new seq with only the prefill token).
    dict(B=2, H=4, d=64,  context_lens=[1, 5], block_size=16),
    # Non-power-of-2 head dim? Skip — real models always use powers of 2.
]


def run_one_case(case: dict, seed: int = 0, dtype: torch.dtype = torch.float32):
    torch.manual_seed(seed)
    B = case["B"]
    H = case["H"]
    d = case["d"]
    block_size = case["block_size"]
    context_lens_py = case["context_lens"]
    N_max = max(context_lens_py)

    device = "cpu"  # Phase 0: CPU fp32 for tight tolerance.

    # Random Q and K/V of maximum length. We'll only read up to context_lens[b]
    # in each sequence; tokens past that are ignored by both references.
    Q = torch.randn(B, H, d, dtype=dtype, device=device)
    K = torch.randn(B, H, N_max, d, dtype=dtype, device=device)
    V = torch.randn(B, H, N_max, d, dtype=dtype, device=device)
    context_lens = torch.tensor(context_lens_py, dtype=torch.int32, device=device)

    scale = 1.0 / (d ** 0.5)

    # Ground truth: naive contiguous attention with context_len mask.
    out_naive = naive_decode_attention(Q, K, V, context_lens=context_lens, scale=scale)

    # Paged path.
    K_cache, V_cache, block_table, ctx_out = pack_kv_paged(K, V, block_size, context_lens)
    assert torch.equal(ctx_out, context_lens)  # pack should preserve ctx lens

    out_paged = paged_attention_ref(Q, K_cache, V_cache, block_table, context_lens, scale=scale)

    assert out_paged.shape == out_naive.shape == (B, H, d), (
        f"shape mismatch: naive {out_naive.shape} paged {out_paged.shape}")

    max_abs_diff = (out_naive - out_paged).abs().max().item()
    max_rel_diff = ((out_naive - out_paged).abs() /
                    (out_naive.abs() + 1e-8)).max().item()

    # fp32 CPU reference should be exact to float precision.
    ok = torch.allclose(out_naive, out_paged, rtol=1e-5, atol=1e-5)

    return {
        "ok": ok,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "case": case,
    }


def main():
    print("Lesson 11 · Phase 0 — paged reference vs contiguous attention")
    print("=" * 72)

    all_pass = True
    for i, case in enumerate(CASES):
        r = run_one_case(case)
        status = "PASS" if r["ok"] else "FAIL"
        print(
            f"[{status}] case {i:2d}: B={case['B']} H={case['H']} d={case['d']:3d} "
            f"ctx={case['context_lens']} block={case['block_size']:3d}  "
            f"abs_diff={r['max_abs_diff']:.2e}  rel_diff={r['max_rel_diff']:.2e}"
        )
        if not r["ok"]:
            all_pass = False

    # Sweep: for fixed shape, try block_sizes {8, 16, 32, 64} — output must
    # be identical regardless of packing granularity.
    print()
    print("Sweep: block_size invariance (same Q/K/V, different block_size)")
    print("-" * 72)

    torch.manual_seed(42)
    B, H, d, N = 2, 4, 64, 96
    Q = torch.randn(B, H, d, dtype=torch.float32)
    K = torch.randn(B, H, N, d, dtype=torch.float32)
    V = torch.randn(B, H, N, d, dtype=torch.float32)
    ctx = torch.tensor([N, N - 7], dtype=torch.int32)  # one full, one partial

    scale = 1.0 / (d ** 0.5)
    out_naive = naive_decode_attention(Q, K, V, context_lens=ctx, scale=scale)

    outs_by_bs = {}
    for bs in (8, 16, 32, 64):
        K_cache, V_cache, bt, _ = pack_kv_paged(K, V, bs, ctx)
        out = paged_attention_ref(Q, K_cache, V_cache, bt, ctx, scale=scale)
        diff_vs_naive = (out - out_naive).abs().max().item()
        outs_by_bs[bs] = out
        print(f"  block_size={bs:3d}: diff vs naive = {diff_vs_naive:.2e}")
        if not torch.allclose(out, out_naive, rtol=1e-5, atol=1e-5):
            all_pass = False
            print("    FAIL")

    # Pairwise: all paged outputs should be identical across block_size.
    ref_bs = 16
    for bs, out in outs_by_bs.items():
        if bs == ref_bs:
            continue
        d_vs_ref = (out - outs_by_bs[ref_bs]).abs().max().item()
        print(f"  block_size={bs:3d} vs block_size={ref_bs}: diff = {d_vs_ref:.2e}")
        if d_vs_ref > 1e-5:
            all_pass = False

    print()
    print("=" * 72)
    if all_pass:
        print("Phase 0 OK — paged reference is correct against contiguous attention.")
        return 0
    else:
        print("Phase 0 FAIL — see diffs above.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
