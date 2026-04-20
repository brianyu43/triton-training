"""
Lesson 09 · Phase 3 · MHA Flash Attention speed benchmark.

3-way comparison on LLaMA-like shapes:
  - ours : triton_flash_attention_mha (our kernel, autotuned)
  - sdpa : F.scaled_dot_product_attention (torch picks backend; FA-2 on sm_80+)
  - naive: (Q @ K^T * scale).softmax(-1) @ V, with causal mask if requested

Metrics: median latency (ms), achieved TFLOPS, speedup vs SDPA.

Causal FLOP accounting halves the count since roughly half the score matrix
is masked out (the FA-v2 loop skip realizes this saving in wall time).

Usage (on the L4 VM after repo sync):
    python3 triton_kernels/bench/bench_flash_attention_mha_speed.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton.testing

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from triton_kernels.flash_attention_mha import (  # noqa: E402
    autotuned_best_config_str,
    triton_flash_attention_mha,
)


def naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    is_causal: bool) -> torch.Tensor:
    """Plain (Q @ K^T).softmax @ V, fp32 softmax for stability in fp16 inputs."""
    scale = 1.0 / (q.shape[-1] ** 0.5)
    # Upcast for softmax to avoid fp16 overflow on large N.
    s = (q.float() @ k.float().transpose(-2, -1)) * scale
    if is_causal:
        N = q.shape[-2]
        mask = torch.ones(N, N, device=q.device, dtype=torch.bool).tril()
        s = s.masked_fill(~mask, float("-inf"))
    p = s.softmax(dim=-1)
    out = p @ v.float()
    return out.to(q.dtype)


def attention_flops(B: int, H: int, N: int, d: int, is_causal: bool) -> float:
    """Total FLOPs for full attention fwd. Two matmuls: Q@K^T and P@V.

    Each matmul is 2 * B * H * N * N * d FLOPs (multiply + add).
    Softmax cost is O(B*H*N*N) — small next to the matmuls, ignored.
    Causal halves the effective FLOPs (upper triangle would be skipped).
    """
    f = 4.0 * B * H * N * N * d
    if is_causal:
        f *= 0.5
    return f


def bench_one(B: int, H: int, N: int, d: int, dtype: torch.dtype,
              is_causal: bool, include_naive: bool = True) -> None:
    torch.manual_seed(B * 1_000_003 + H * 10_007 + N * 131 + d)
    q = torch.randn(B, H, N, d, device="cuda", dtype=dtype)
    k = torch.randn(B, H, N, d, device="cuda", dtype=dtype)
    v = torch.randn(B, H, N, d, device="cuda", dtype=dtype)

    def run_ours():   return triton_flash_attention_mha(q, k, v, is_causal=is_causal)
    def run_sdpa():   return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    def run_naive():  return naive_attention(q, k, v, is_causal=is_causal)

    # Trigger autotune once so the first timed iter isn't an outlier.
    _ = run_ours()
    torch.cuda.synchronize()

    ms_ours = triton.testing.do_bench(run_ours, warmup=25, rep=100)
    ms_sdpa = triton.testing.do_bench(run_sdpa, warmup=25, rep=100)

    # Naive builds an (N, N) score matrix — skip when it would OOM or dominate time.
    naive_mem_bytes = B * H * N * N * 4  # fp32 scores
    naive_fits = naive_mem_bytes < (4 << 30)  # <4 GB
    if include_naive and naive_fits:
        ms_naive = triton.testing.do_bench(run_naive, warmup=5, rep=20)
        naive_str = f"{ms_naive:7.3f}ms ({attention_flops(B,H,N,d,is_causal)/(ms_naive*1e-3)/1e12:5.1f}TF)"
    else:
        ms_naive = float("nan")
        naive_str = "          (skipped)"

    f = attention_flops(B, H, N, d, is_causal)
    def tf(ms: float) -> float:
        return f / (ms * 1e-3) / 1e12 if ms > 0 else 0.0

    speedup_vs_sdpa = ms_sdpa / ms_ours
    ctag = "causal" if is_causal else "  full"
    print(
        f"  B={B} H={H:>2} N={N:>4} d={d:>3} {ctag}  "
        f"ours={ms_ours:7.3f}ms ({tf(ms_ours):5.1f}TF)  "
        f"sdpa={ms_sdpa:7.3f}ms ({tf(ms_sdpa):5.1f}TF)  "
        f"naive={naive_str}  "
        f"ours/sdpa={speedup_vs_sdpa:.2f}×"
    )


def main() -> None:
    dev = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"device = {dev}   cap = sm_{cap[0]}{cap[1]}")
    print(f"torch  = {torch.__version__}")

    # Fair fp32 comparison convention (from lesson 08).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("\n=== LLaMA-7B-like shapes, fp16, CAUSAL ===")
    print("(d=128 matches LLaMA-7B per-head dim; H=32 matches LLaMA-7B heads)")
    for (B, H, N, d) in [
        (1, 32,  512, 128),
        (1, 32, 1024, 128),
        (1, 32, 2048, 128),
        (1, 32, 4096, 128),
        (2, 32, 1024, 128),
        (2, 32, 2048, 128),
        (4, 32, 1024, 128),
    ]:
        bench_one(B, H, N, d, torch.float16, is_causal=True, include_naive=True)

    print("\n=== non-causal (control — compare causal speedup against) ===")
    for (B, H, N, d) in [
        (1, 32, 1024, 128),
        (2, 32, 2048, 128),
    ]:
        bench_one(B, H, N, d, torch.float16, is_causal=False, include_naive=True)

    print("\n=== GPT-2-like short shapes (B>=8, smaller d) ===")
    for (B, H, N, d) in [
        (8, 12, 1024, 64),
        (16, 12, 512, 64),
    ]:
        bench_one(B, H, N, d, torch.float16, is_causal=True, include_naive=False)

    print(f"\nautotune picked for last shape: {autotuned_best_config_str()}")


if __name__ == "__main__":
    main()
