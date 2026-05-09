from __future__ import annotations

import argparse

import torch
import triton
import triton.language as tl


SHAPES = [
    (4096, 5120, 4096),
    (2048, 3072, 2048),
]


def make_configs():
    configs = []
    for warps in (4, 8):
        for stages in (2, 3, 4):
            for group_m in (4, 8, 16, 32):
                configs.append(
                    (
                        f"128x128x32_w{warps}_s{stages}_g{group_m}",
                        128,
                        128,
                        32,
                        warps,
                        stages,
                        group_m,
                    )
                )
    for bm, bn in ((64, 128), (128, 64), (64, 256), (256, 64)):
        for warps in (4, 8):
            for stages in (3, 4):
                configs.append(
                    (
                        f"{bm}x{bn}x32_w{warps}_s{stages}_g8",
                        bm,
                        bn,
                        32,
                        warps,
                        stages,
                        8,
                    )
                )
    return configs


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
    dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda")
    dummy.fill_(42)
    del dummy


def time_us(fn, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        clear_l2()
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return sum(times) / len(times), min(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
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
        for name, bm, bn, bk, warps, stages, group_m in make_configs():
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
                    GROUP_M=group_m,
                    num_warps=warps,
                    num_stages=stages,
                )
                fn()
                torch.cuda.synchronize()
                ok = torch.allclose(c, ref, rtol=1e-5, atol=1e-8)
                diff = (c.float() - ref.float()).abs().max().item()
                mean, best = time_us(fn, args.warmup, args.iters)
                print(f"{m}x{n}x{k},{name},{ok},{diff:.6g},{mean:.3f},{best:.3f}")
            except Exception as exc:
                print(f"{m}x{n}x{k},{name},ERROR,{type(exc).__name__}:{exc},nan,nan")


if __name__ == "__main__":
    main()
