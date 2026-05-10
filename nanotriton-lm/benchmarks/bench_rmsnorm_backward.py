from __future__ import annotations

import argparse
import json

from benchmarks.utils import time_callable
from nanotriton.autograd.rmsnorm_fn import triton_rmsnorm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Triton RMSNorm forward+backward against PyTorch reference.")
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
    grad_out = torch.randn_like(x)

    def torch_ref():
        x_ref = x.detach().clone().requires_grad_(True)
        weight_ref = weight.detach().clone().requires_grad_(True)
        x_float = x_ref.float()
        y = (x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * weight_ref.float()).to(
            dtype=dtype
        )
        y.backward(grad_out)
        return x_ref.grad, weight_ref.grad

    def triton_kernel():
        x_ref = x.detach().clone().requires_grad_(True)
        weight_ref = weight.detach().clone().requires_grad_(True)
        y = triton_rmsnorm(x_ref, weight_ref)
        y.backward(grad_out)
        return x_ref.grad, weight_ref.grad

    expected_dx, expected_dw = torch_ref()
    actual_dx, actual_dw = triton_kernel()
    torch.testing.assert_close(actual_dx, expected_dx, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(actual_dw, expected_dw, atol=3e-3, rtol=3e-3)

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
