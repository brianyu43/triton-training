from __future__ import annotations

import argparse
import json

from benchmarks.utils import time_callable
from nanotriton.autograd.swiglu_fn import triton_swiglu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Triton SwiGLU forward+backward against PyTorch reference.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=512)
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
    a = torch.randn((args.batch, args.seq, args.hidden), device="cuda", dtype=dtype)
    b = torch.randn_like(a)
    grad_out = torch.randn_like(a)

    def torch_ref():
        a_ref = a.detach().clone().requires_grad_(True)
        b_ref = b.detach().clone().requires_grad_(True)
        y = torch.nn.functional.silu(a_ref.float()).to(dtype=dtype) * b_ref
        y.backward(grad_out)
        return a_ref.grad, b_ref.grad

    def triton_kernel():
        a_ref = a.detach().clone().requires_grad_(True)
        b_ref = b.detach().clone().requires_grad_(True)
        y = triton_swiglu(a_ref, b_ref)
        y.backward(grad_out)
        return a_ref.grad, b_ref.grad

    expected_da, expected_db = torch_ref()
    actual_da, actual_db = triton_kernel()
    torch.testing.assert_close(actual_da, expected_da, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(actual_db, expected_db, atol=3e-3, rtol=3e-3)

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
