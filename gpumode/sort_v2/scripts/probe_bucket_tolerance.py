#!/usr/bin/env python3
import argparse
import math

import torch


DEFAULT_SIZES = [100000, 500000, 1000000]
DEFAULT_SEEDS = [6252, 19252, 54352]
MAX_P99_ITEMS = 5_000_000


def generate_input(size: int, seed: int) -> torch.Tensor:
    rows = int(size**0.5)
    cols = (size + rows - 1) // rows
    gen = torch.Generator(device="cuda")
    result = torch.empty((rows, cols), device="cuda", dtype=torch.float32)
    for i in range(rows):
        row_seed = seed + i
        gen.manual_seed(row_seed)
        result[i, :] = torch.randn(cols, device="cuda", dtype=torch.float32, generator=gen) + row_seed
    return result.flatten()[:size].contiguous()


def approx_bucket_order(values: torch.Tensor, bpu: int, bias: int) -> tuple[torch.Tensor, int, int, int]:
    n = values.numel()
    rows = int(math.sqrt(n))
    cols = (n + rows - 1) // rows
    base = int(torch.round(values[:cols].mean()).item())
    bucket_count = (rows + 2 * bias) * bpu
    raw = torch.floor((values - base + bias) * bpu).to(torch.int64)
    under = int((raw < 0).sum().item())
    over = int((raw >= bucket_count).sum().item())
    keys = torch.clamp(raw, 0, bucket_count - 1).to(torch.int32)
    order = torch.argsort(keys, stable=False)
    return values[order].contiguous(), base, under, over


def p99_abs_error(diff: torch.Tensor) -> float:
    if diff.numel() <= MAX_P99_ITEMS:
        sample = diff
    else:
        stride = math.ceil(diff.numel() / MAX_P99_ITEMS)
        sample = diff[::stride]
    return float(torch.quantile(sample, 0.99).item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--bpus", nargs="+", type=int, default=[128, 64])
    parser.add_argument("--biases", nargs="+", type=int, default=[16, 24])
    args = parser.parse_args()

    torch.cuda.init()
    print(f"device={torch.cuda.get_device_name(0)}")
    print("size,seed,bpu,bias,base,under,over,allclose,max_abs,p99_abs")

    for size in args.sizes:
        for seed in args.seeds:
            values = generate_input(size, seed)
            ref = torch.sort(values)[0]
            for bpu in args.bpus:
                for bias in args.biases:
                    out, base, under, over = approx_bucket_order(values, bpu=bpu, bias=bias)
                    diff = (out - ref).abs()
                    ok = torch.allclose(out, ref, rtol=1e-5, atol=1e-8)
                    max_abs = float(diff.max().item())
                    p99_abs = p99_abs_error(diff)
                    print(f"{size},{seed},{bpu},{bias},{base},{under},{over},{ok},{max_abs:.8f},{p99_abs:.8f}", flush=True)
            del values, ref
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
