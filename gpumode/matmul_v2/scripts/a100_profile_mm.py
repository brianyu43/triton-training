#!/usr/bin/env python3
import argparse
import statistics

import torch


SHAPES = {
    "final": (4096, 5120, 4096),
    "mid": (2048, 3072, 2048),
}


def run_variant(name, a, b, c):
    if name == "mm":
        torch.mm(a, b, out=c)
        return c
    if name == "return_mm":
        return torch.mm(a, b, out=c)
    if name == "matmul":
        return torch.matmul(a, b, out=c)
    if name == "addmm":
        return torch.addmm(c, a, b, beta=0.0, alpha=1.0, out=c)
    if name == "aten_mm":
        return torch.ops.aten.mm.out(a, b, out=c)
    raise ValueError(f"unknown variant: {name}")


def time_variant(name, a, b, c, warmup, iters):
    for _ in range(warmup):
        run_variant(name, a, b, c)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.nvtx.range_push(name)
        start.record()
        out = run_variant(name, a, b, c)
        end.record()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        if out.data_ptr() != c.data_ptr():
            raise RuntimeError(f"{name} did not return the provided output tensor")
        times.append(start.elapsed_time(end) * 1000.0)
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=sorted(SHAPES), default="final")
    parser.add_argument(
        "--variant",
        choices=["all", "mm", "return_mm", "matmul", "addmm", "aten_mm"],
        default="all",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123456)
    args = parser.parse_args()

    m, n, k = SHAPES[args.shape]
    gen = torch.Generator(device="cuda")
    gen.manual_seed(args.seed)
    a = torch.randn((m, k), device="cuda", dtype=torch.float16, generator=gen)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16, generator=gen)
    c = torch.empty((m, n), device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()

    variants = ["mm", "return_mm", "matmul", "addmm", "aten_mm"]
    if args.variant != "all":
        variants = [args.variant]

    print(f"device,{torch.cuda.get_device_name()}")
    print(f"torch,{torch.__version__}")
    print(f"shape,{m},{n},{k}")
    print("variant,mean_us,median_us,best_us,worst_us,std_us,iters")
    for variant in variants:
        times = time_variant(variant, a, b, c, args.warmup, args.iters)
        mean = statistics.fmean(times)
        median = statistics.median(times)
        best = min(times)
        worst = max(times)
        std = statistics.pstdev(times) if len(times) > 1 else 0.0
        print(
            f"{variant},{mean:.3f},{median:.3f},{best:.3f},"
            f"{worst:.3f},{std:.3f},{len(times)}"
        )


if __name__ == "__main__":
    main()
