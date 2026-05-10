from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import types
from pathlib import Path

import torch
import triton


def _install_task_stub() -> None:
    task = types.ModuleType("task")
    task.input_t = tuple
    task.output_t = torch.Tensor
    sys.modules["task"] = task


def _load_submission(path: Path):
    _install_task_stub()
    spec = importlib.util.spec_from_file_location("submission", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["submission"] = module
    spec.loader.exec_module(module)
    return module


def time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        best = min(best, start.elapsed_time(end))
    return best


def gbps(size: int, ms: float) -> float:
    # Two fp16 reads plus one fp16 write.
    bytes_moved = 3 * size * size * 2
    return bytes_moved / (ms * 1e-3) / 1e9


def make_inputs(size: int, seed: int):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    a = torch.randn(size, size, device="cuda", dtype=torch.float16, generator=gen).contiguous()
    b = torch.randn(size, size, device="cuda", dtype=torch.float16, generator=gen).contiguous()
    out = torch.empty(size, size, device="cuda", dtype=torch.float16).contiguous()
    return a, b, out


def check_once(submission, size: int) -> None:
    a, b, out = make_inputs(size, 123)
    got = submission.custom_kernel((a, b, out))
    ref = a + b
    torch.cuda.synchronize()
    max_err = (got - ref).abs().max().item()
    ok = max_err == 0.0
    print(f"correctness size={size}: max_err={max_err:.3e} ok={ok}")
    if not ok:
        raise AssertionError(f"mismatch for size {size}")


def bench_submission(submission, sizes: list[int], warmup: int, iters: int) -> None:
    print("mode,size,best_us,gbps")
    for size in sizes:
        a, b, out = make_inputs(size, size)
        fn = lambda: submission.custom_kernel((a, b, out))
        ms = time_ms(fn, warmup, iters)
        print(f"submission,{size},{ms * 1000:.3f},{gbps(size, ms):.3f}")


def bench_torch(sizes: list[int], warmup: int, iters: int) -> None:
    for size in sizes:
        a, b, out = make_inputs(size, size + 1)
        fn = lambda: torch.add(a, b, out=out)
        ms = time_ms(fn, warmup, iters)
        print(f"torch_add_out,{size},{ms * 1000:.3f},{gbps(size, ms):.3f}")


def sweep(submission, size: int, warmup: int, iters: int) -> None:
    a, b, out = make_inputs(size, 999)
    print("mode,size,block,warps,best_us,gbps")
    for block in [256, 512, 1024, 2048, 4096, 8192]:
        grid = (triton.cdiv(a.numel(), block),)
        for warps in [4, 8]:
            if block in (256, 1024) and warps == 8:
                continue
            fn = lambda block=block, grid=grid, warps=warps: submission._add_kernel[
                grid
            ](a, b, out, BLOCK_SIZE=block, num_warps=warps)
            ms = time_ms(fn, warmup, iters)
            print(f"sweep,{size},{block},{warps},{ms * 1000:.3f},{gbps(size, ms):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, default=Path("submission.py"))
    parser.add_argument("--sizes", default="1024,2048,4096,8192,16384")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--sweep-size", type=int, default=16384)
    args = parser.parse_args()

    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    print("triton", triton.__version__)

    submission = _load_submission(args.submission)
    sizes = [int(x) for x in args.sizes.split(",") if x]
    check_once(submission, min(sizes))
    bench_submission(submission, sizes, args.warmup, args.iters)
    bench_torch(sizes, args.warmup, args.iters)
    sweep(submission, args.sweep_size, args.warmup, args.iters)


if __name__ == "__main__":
    main()
