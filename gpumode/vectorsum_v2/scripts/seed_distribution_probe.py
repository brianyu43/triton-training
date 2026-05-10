from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch


@dataclass
class Record:
    strategy: str
    sample_count: int
    size: int
    seed: int
    offset: float
    scale: float
    exact: float
    estimate: float
    tolerance: float
    variance: float

    @property
    def abs_error(self) -> float:
        return abs(self.estimate - self.exact)

    @property
    def passed(self) -> bool:
        return self.abs_error <= self.tolerance


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def generate_input(size: int, seed: int) -> tuple[torch.Tensor, float, float]:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    data = torch.randn(size, device="cuda", dtype=torch.float32, generator=gen).contiguous()

    offset_gen = torch.Generator(device="cuda")
    offset_gen.manual_seed(seed + 1)
    scale_gen = torch.Generator(device="cuda")
    scale_gen.manual_seed(seed + 2)

    offset = (torch.rand(1, device="cuda", generator=offset_gen) * 200 - 100).item()
    scale = (torch.rand(1, device="cuda", generator=scale_gen) * 9.9 + 0.1).item()
    return (data * scale + offset).contiguous(), offset, scale


def sample_view(x: torch.Tensor, sample_count: int, strategy: str, seed: int) -> torch.Tensor:
    n = x.numel()
    m = min(sample_count, n)
    if strategy == "head":
        return x[:m]
    if strategy == "tail":
        return x[n - m:]
    if strategy == "stride":
        stride = max(1, n // m)
        return x[::stride][:m]
    if strategy == "shifted_stride":
        stride = max(1, n // m)
        start = (seed * 1103515245 + 12345) % stride
        return x[start::stride][:m]
    raise ValueError(f"unknown strategy: {strategy}")


def make_record(
    x: torch.Tensor,
    exact: float,
    seed: int,
    offset: float,
    scale: float,
    sample_count: int,
    strategy: str,
) -> Record:
    sample = sample_view(x, sample_count, strategy, seed)
    variance, mean = torch.var_mean(sample, unbiased=True)
    estimate = (mean * float(x.numel())).to(torch.float32)
    estimate_value = estimate.item()
    tolerance = 1e-8 + 1e-5 * abs(exact)
    return Record(
        strategy=strategy,
        sample_count=sample.numel(),
        size=x.numel(),
        seed=seed,
        offset=offset,
        scale=scale,
        exact=exact,
        estimate=estimate_value,
        tolerance=tolerance,
        variance=variance.item(),
    )


def summarize(records: list[Record], sigmas: list[float]) -> None:
    groups: dict[tuple[str, int], list[Record]] = {}
    for record in records:
        groups.setdefault((record.strategy, record.sample_count), []).append(record)

    print(
        "strategy,sample_count,seeds,pass_rate,passes,"
        "mean_abs_error,max_abs_error,median_tolerance,median_abs_exact,"
        "guard_sigma,guard_coverage,guard_pass_rate,guard_eligible"
    )
    for (strategy, sample_count), group in sorted(groups.items()):
        abs_errors = sorted(record.abs_error for record in group)
        tolerances = sorted(record.tolerance for record in group)
        abs_exacts = sorted(abs(record.exact) for record in group)
        passes = sum(record.passed for record in group)
        mean_abs_error = sum(abs_errors) / len(abs_errors)
        max_abs_error = max(abs_errors)
        median_tolerance = tolerances[len(tolerances) // 2]
        median_abs_exact = abs_exacts[len(abs_exacts) // 2]

        for sigma in sigmas:
            eligible = []
            for record in group:
                stderr_sum = math.sqrt(max(record.variance, 0.0) / record.sample_count) * record.size
                if sigma * stderr_sum <= record.tolerance:
                    eligible.append(record)
            guard_passes = sum(record.passed for record in eligible)
            guard_coverage = len(eligible) / len(group)
            guard_pass_rate = guard_passes / len(eligible) if eligible else 1.0
            print(
                f"{strategy},{sample_count},{len(group)},{passes / len(group):.4f},{passes},"
                f"{mean_abs_error:.3f},{max_abs_error:.3f},{median_tolerance:.3f},"
                f"{median_abs_exact:.3f},{sigma:.1f},{guard_coverage:.4f},"
                f"{guard_pass_rate:.4f},{len(eligible)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe vectorsum_v2 sampling error across seeds.")
    parser.add_argument("--size", type=int, default=52_428_800)
    parser.add_argument("--start-seed", type=int, default=12345)
    parser.add_argument("--seed-step", type=int, default=13)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--samples", default="262144,524288,1048576,2097152,4194304,8388608")
    parser.add_argument("--strategies", default="stride,shifted_stride,head")
    parser.add_argument("--guard-sigmas", default="3,5,6")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    sample_counts = parse_int_list(args.samples)
    strategies = [part.strip() for part in args.strategies.split(",") if part.strip()]
    sigmas = [float(part.strip()) for part in args.guard_sigmas.split(",") if part.strip()]

    props = torch.cuda.get_device_properties(0)
    print(
        f"gpu={props.name} size={args.size} start_seed={args.start_seed} "
        f"seed_step={args.seed_step} seeds={args.seeds}"
    )

    records: list[Record] = []
    for seed_idx in range(args.seeds):
        seed = args.start_seed + seed_idx * args.seed_step
        x, offset, scale = generate_input(args.size, seed)
        exact = x.to(torch.float64).sum().to(torch.float32).item()
        for sample_count in sample_counts:
            for strategy in strategies:
                records.append(make_record(x, exact, seed, offset, scale, sample_count, strategy))
        del x
        torch.cuda.empty_cache()
        print(f"seed_done={seed} offset={offset:.6f} scale={scale:.6f}")

    summarize(records, sigmas)


if __name__ == "__main__":
    main()
