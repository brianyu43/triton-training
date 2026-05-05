from __future__ import annotations

import argparse
import math

import torch
import triton
import triton.language as tl


SHAPES = [
    (2048, 2048, 2048),
    (2048, 3072, 2048),
    (4096, 5120, 4096),
    (1024, 1024, 1024),
    (1024, 1536, 1024),
]

CONFIGS = [
    ("64x64x64_w4_s4", 64, 64, 64, 4, 4),
    ("64x128x64_w4_s4", 64, 128, 64, 4, 4),
    ("128x128x64_w4_s4", 128, 128, 64, 4, 4),
    ("128x128x32_w4_s5", 128, 128, 32, 4, 5),
    ("128x128x64_w8_s4", 128, 128, 64, 8, 4),
    ("128x256x64_w8_s4", 128, 256, 64, 8, 4),
]


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m: tl.constexpr = M // BM
    num_pid_n: tl.constexpr = N // BN

    group_id = pid // (GROUP_M * num_pid_n)
    first_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_m, GROUP_M)
    pid_in_group = pid % (GROUP_M * num_pid_n)
    pid_m = first_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptr + offs_m[:, None] * K + (k0 + offs_k)[None, :])
        b = tl.load(b_ptr + (k0 + offs_k)[:, None] * N + offs_n[None, :])
        acc = tl.dot(a, b, acc)

    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16))


def make_inputs(m: int, n: int, k: int, seed: int):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    a = torch.empty(m, k, device="cuda", dtype=torch.float16)
    a.uniform_(0, 1, generator=gen)
    b = torch.empty(k, n, device="cuda", dtype=torch.float16)
    b.uniform_(0, 1, generator=gen)
    c = torch.empty(m, n, device="cuda", dtype=torch.float16)
    return a, b, c


def clear_l2():
    # Slightly larger than L4 L2. The official harness uses the same idea.
    dummy = torch.empty(64 * 1024 * 1024, device="cuda", dtype=torch.int8)
    dummy.zero_()


def time_us(fn, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        clear_l2()
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return sum(times) / len(times), min(times)


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123456)
    args = parser.parse_args()

    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    print("torch", torch.__version__, "triton", triton.__version__)
    print("shape,config,ok,max_abs_diff,mean_us,best_us")

    for m, n, k in SHAPES:
        a, b, c = make_inputs(m, n, k, args.seed)
        ref = torch.empty_like(c)
        torch.mm(a, b, out=ref)
        torch.cuda.synchronize()

        for name, bm, bn, bk, warps, stages in CONFIGS:
            if m % bm or n % bn or k % bk:
                continue
            c.zero_()
            grid = ((m // bm) * (n // bn),)
            try:
                fn = lambda: _matmul_kernel[grid](
                    a,
                    b,
                    c,
                    M=m,
                    N=n,
                    K=k,
                    BM=bm,
                    BN=bn,
                    BK=bk,
                    GROUP_M=8,
                    num_warps=warps,
                    num_stages=stages,
                )
                fn()
                torch.cuda.synchronize()
                ok = torch.allclose(c, ref, rtol=1e-5, atol=1e-8)
                diff = max_diff(c, ref)
                mean, best = time_us(fn, args.warmup, args.iters)
                print(f"{m}x{n}x{k},{name},{ok},{diff:.6g},{mean:.3f},{best:.3f}")
            except Exception as exc:
                print(f"{m}x{n}x{k},{name},ERROR,{type(exc).__name__}:{exc},nan,nan")


if __name__ == "__main__":
    main()
