"""
Lesson 11 · Phase 3 · ncu drill driver.

Single shape, one impl per run. ncu will catch the warmup-skipped launches.

  --mode paged : triton_paged_attention_decode (our kernel)
  --mode sdpa  : torch.nn.functional.scaled_dot_product_attention (contig)

We deliberately do NOT loop inside Python so `--iterations` directly
controls the number of kernel launches ncu sees.

--list-kernels prints the actual kernel names the dispatcher launches so
we can feed the right regex to ncu.

Usage under ncu (example — Lesson 10 toolchain):

    sudo -E ~/miniforge3/envs/drug_discovery/bin/python \\
        triton_kernels/bench/lesson11_ncu_profile.py \\
        --mode paged --block-size 16 --warmup 10 --iterations 5

    sudo -E ncu --launch-skip 10 --launch-count 5 \\
        --section SpeedOfLight --section WarpStateStats \\
        --section LaunchStats --section MemoryWorkloadAnalysis \\
        python3 triton_kernels/bench/lesson11_ncu_profile.py \\
        --mode paged --block-size 16 --warmup 10 --iterations 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from triton_kernels.paged_attention import triton_paged_attention_decode   # noqa: E402
from triton_kernels.paged_attention_ref import pack_kv_paged                # noqa: E402


def build_inputs(B: int, H: int, H_kv: int, d: int, ctx: int,
                 block_size: int, dtype: torch.dtype):
    torch.manual_seed(0)
    device = "cuda"
    Q = torch.randn(B, H, d, dtype=dtype, device=device)
    K = torch.randn(B, H_kv, ctx, d, dtype=dtype, device=device)
    V = torch.randn(B, H_kv, ctx, d, dtype=dtype, device=device)
    ctx_t = torch.full((B,), ctx, dtype=torch.int32, device=device)
    K_cache, V_cache, block_table, _ = pack_kv_paged(K, V, block_size, ctx_t)
    return Q, K, V, K_cache, V_cache, block_table, ctx_t


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["paged", "sdpa"], required=True)
    p.add_argument("--B", type=int, default=8)
    p.add_argument("--H", type=int, default=32)
    p.add_argument("--H-kv", type=int, default=32,
                   help="Kv heads (==H for MHA; < H for GQA)")
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--list-kernels", action="store_true",
                   help="run one iteration under torch.profiler, print kernel names")
    args = p.parse_args()

    Q, K, V, K_cache, V_cache, bt, ctx_t = build_inputs(
        args.B, args.H, args.H_kv, args.d, args.ctx,
        args.block_size, torch.float16)

    scale = 1.0 / (args.d ** 0.5)

    def run():
        if args.mode == "paged":
            return triton_paged_attention_decode(
                Q, K_cache, V_cache, bt, ctx_t, scale=scale
            )
        # SDPA baseline.
        Q4 = Q.unsqueeze(2)   # (B, H, 1, d)
        if args.H_kv == args.H:
            return F.scaled_dot_product_attention(Q4, K, V, is_causal=False, scale=scale)
        return F.scaled_dot_product_attention(
            Q4, K, V, is_causal=False, scale=scale, enable_gqa=True
        )

    # Warmup: Triton autotune compile, cuDNN algorithm pick, cache warmup.
    for _ in range(args.warmup):
        _ = run()
    torch.cuda.synchronize()

    if args.list_kernels:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            _ = run()
            torch.cuda.synchronize()
        events = prof.key_averages()

        def dev_time(e):
            for attr in ("device_time_total", "cuda_time_total",
                         "self_device_time_total"):
                if hasattr(e, attr):
                    return getattr(e, attr)
            return 0.0
        cuda_events = [e for e in events if dev_time(e) > 0]
        print(f"\n=== CUDA kernels launched for mode={args.mode} ===")
        for e in sorted(cuda_events, key=lambda x: -dev_time(x)):
            print(f"  {dev_time(e)/1e3:8.1f}us  {e.key}")
        return

    # Steady-state launches. ncu --launch-skip/count targets these.
    for _ in range(args.iterations):
        _ = run()
    torch.cuda.synchronize()

    print(f"done mode={args.mode} B={args.B} H={args.H} H_kv={args.H_kv} "
          f"d={args.d} ctx={args.ctx} block_size={args.block_size} "
          f"warmup={args.warmup} iters={args.iterations}")


if __name__ == "__main__":
    main()
