"""
Lesson 10 · Phase 3 · minimal driver for ncu profiling.

Runs ONE of two implementations at a fixed (B, H, N, d) shape so that
`ncu --launch-skip N --launch-count 1` captures a single clean forward pass:

  - --mode ours  : our Triton flash_attention_mha_fwd_kernel
  - --mode sdpa  : torch.nn.functional.scaled_dot_product_attention
                   (on L4 + fp16 + causal, this dispatches to a cuDNN /
                   aten attention kernel — we profile whichever the
                   dispatcher picks)

We deliberately do NOT loop inside Python so `--iterations` directly
controls the number of kernel launches ncu sees.

We ALSO support --list-kernels which uses torch.profiler to print the
actual kernel names launched, so we can feed the right regex to ncu.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from triton_kernels.flash_attention_mha import (  # noqa: E402
    triton_flash_attention_mha,
)


def build_inputs(B: int, H: int, N: int, d: int, dtype: torch.dtype):
    torch.manual_seed(0)
    q = torch.randn(B, H, N, d, device="cuda", dtype=dtype)
    k = torch.randn(B, H, N, d, device="cuda", dtype=dtype)
    v = torch.randn(B, H, N, d, device="cuda", dtype=dtype)
    return q, k, v


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ours", "sdpa"], required=True)
    p.add_argument("--B", type=int, default=1)
    p.add_argument("--H", type=int, default=32)
    p.add_argument("--N", type=int, default=2048)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--no-causal", action="store_true",
                   help="disable causal mask (default: causal on)")
    p.add_argument("--warmup", type=int, default=20,
                   help="warmup launches (use with ncu --launch-skip)")
    p.add_argument("--iterations", type=int, default=1,
                   help="steady-state launches (use with ncu --launch-count)")
    p.add_argument("--list-kernels", action="store_true",
                   help="run under torch.profiler and print kernel names")
    args = p.parse_args()

    causal = not args.no_causal
    q, k, v = build_inputs(args.B, args.H, args.N, args.d, torch.float16)

    def run():
        if args.mode == "ours":
            return triton_flash_attention_mha(q, k, v, is_causal=causal)
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    # Warmup absorbs Triton autotune + any lazy init + cache warmup.
    for _ in range(args.warmup):
        _ = run()
    torch.cuda.synchronize()

    if args.list_kernels:
        # Run one iteration under profiler, print CUDA kernel names.
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA],
                     record_shapes=False) as prof:
            _ = run()
            torch.cuda.synchronize()
        events = prof.key_averages()
        # torch >= 2.5 renamed cuda_time_total → device_time_total.
        def dev_time(e):
            for attr in ("device_time_total", "cuda_time_total", "self_device_time_total"):
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

    print(f"done mode={args.mode} B={args.B} H={args.H} N={args.N} "
          f"d={args.d} causal={causal} warmup={args.warmup} "
          f"iters={args.iterations}")


if __name__ == "__main__":
    main()
