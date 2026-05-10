from __future__ import annotations

import argparse

import torch
import triton
import triton.language as tl


SHAPE = (4096, 5120, 4096)


@triton.jit
def _matmul_default(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m: tl.constexpr = M // 128
    num_pid_n: tl.constexpr = N // 128
    group_id = pid // (GROUP_M * num_pid_n)
    first_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_m, GROUP_M)
    pid_in_group = pid % (GROUP_M * num_pid_n)
    pid_m = first_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * 128 + tl.arange(0, 128)
    offs_n = pid_n * 128 + tl.arange(0, 128)
    offs_k = tl.arange(0, 32)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, 128), 128)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, 128), 128)
    offs_k = tl.max_contiguous(tl.multiple_of(offs_k, 32), 32)

    acc = tl.zeros((128, 128), tl.float32)
    for k0 in range(0, K, 32):
        a = tl.load(a_ptr + offs_m[:, None] * K + (k0 + offs_k)[None, :])
        b = tl.load(b_ptr + (k0 + offs_k)[:, None] * N + offs_n[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16))


@triton.jit
def _matmul_cache_hint(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m: tl.constexpr = M // 128
    num_pid_n: tl.constexpr = N // 128
    group_id = pid // (GROUP_M * num_pid_n)
    first_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_m, GROUP_M)
    pid_in_group = pid % (GROUP_M * num_pid_n)
    pid_m = first_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * 128 + tl.arange(0, 128)
    offs_n = pid_n * 128 + tl.arange(0, 128)
    offs_k = tl.arange(0, 32)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, 128), 128)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, 128), 128)
    offs_k = tl.max_contiguous(tl.multiple_of(offs_k, 32), 32)

    acc = tl.zeros((128, 128), tl.float32)
    for k0 in range(0, K, 32):
        a = tl.load(
            a_ptr + offs_m[:, None] * K + (k0 + offs_k)[None, :],
            eviction_policy="evict_first",
        )
        b = tl.load(
            b_ptr + (k0 + offs_k)[:, None] * N + offs_n[None, :],
            eviction_policy="evict_last",
        )
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16))


@triton.jit
def _matmul_cg(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m: tl.constexpr = M // 128
    num_pid_n: tl.constexpr = N // 128
    group_id = pid // (GROUP_M * num_pid_n)
    first_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_m, GROUP_M)
    pid_in_group = pid % (GROUP_M * num_pid_n)
    pid_m = first_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * 128 + tl.arange(0, 128)
    offs_n = pid_n * 128 + tl.arange(0, 128)
    offs_k = tl.arange(0, 32)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, 128), 128)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, 128), 128)
    offs_k = tl.max_contiguous(tl.multiple_of(offs_k, 32), 32)

    acc = tl.zeros((128, 128), tl.float32)
    for k0 in range(0, K, 32):
        a = tl.load(
            a_ptr + offs_m[:, None] * K + (k0 + offs_k)[None, :],
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )
        b = tl.load(
            b_ptr + (k0 + offs_k)[:, None] * N + offs_n[None, :],
            cache_modifier=".cg",
            eviction_policy="evict_last",
        )
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16))


def make_inputs(seed: int):
    m, n, k = SHAPE
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
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123456)
    args = parser.parse_args()

    m, n, k = SHAPE
    a, b, c = make_inputs(args.seed)
    ref = torch.empty_like(c)
    torch.mm(a, b, out=ref)
    torch.cuda.synchronize()

    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    print("torch", torch.__version__, "triton", triton.__version__)
    print("variant,group_m,ok,max_abs_diff,mean_us,best_us")

    grid = ((m // 128) * (n // 128),)
    kernels = [
        ("default", _matmul_default),
        ("cache_hint", _matmul_cache_hint),
        ("cg", _matmul_cg),
    ]
    for group_m in (10, 12, 14, 16, 18, 20, 24, 28, 32):
        for name, kernel in kernels:
            try:
                fn = lambda kernel=kernel, group_m=group_m: kernel[grid](
                    a,
                    b,
                    c,
                    M=m,
                    N=n,
                    K=k,
                    GROUP_M=group_m,
                    num_warps=4,
                    num_stages=4,
                )
                fn()
                torch.cuda.synchronize()
                ok = torch.allclose(c, ref, rtol=1e-5, atol=1e-8)
                diff = (c.float() - ref.float()).abs().max().item()
                mean, best = time_us(fn, args.warmup, args.iters)
                print(f"{name},{group_m},{ok},{diff:.6g},{mean:.3f},{best:.3f}")
            except Exception as exc:
                print(f"{name},{group_m},ERROR,{type(exc).__name__}:{exc},nan,nan")


if __name__ == "__main__":
    main()
