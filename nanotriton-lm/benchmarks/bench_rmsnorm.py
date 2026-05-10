from __future__ import annotations

import argparse
import json

from benchmarks.utils import time_callable
from nanotriton.kernels.rmsnorm import rmsnorm_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Triton RMSNorm forward against PyTorch reference.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    dtype = {"float16": torch.float16, "float32": torch.float32}[args.dtype]
    torch.manual_seed(1337)
    x = torch.randn((args.batch, args.seq, args.hidden), device="cuda", dtype=dtype)
    weight = torch.randn((args.hidden,), device="cuda", dtype=dtype)

    def torch_ref():
        x_float = x.float()
        return (x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * weight.float()).to(dtype=dtype)

    def triton_kernel():
        return rmsnorm_forward(x, weight)

    expected = torch_ref()
    actual = triton_kernel()
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)

    torch_timing = time_callable(torch_ref, warmup=args.warmup, iters=args.iters)
    triton_timing = time_callable(triton_kernel, warmup=args.warmup, iters=args.iters)
    print(
        json.dumps(
            {
                "shape": [args.batch, args.seq, args.hidden],
                "dtype": args.dtype,
                "torch_ms": torch_timing.__dict__,
                "triton_ms": triton_timing.__dict__,
                "speedup": torch_timing.median_ms / triton_timing.median_ms,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
