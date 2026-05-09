#!POPCORN leaderboard matmul_v2
#!POPCORN gpus L4

import torch
import triton
import triton.language as tl
from task import input_t, output_t


@triton.jit
def _matmul_128x128x32_default(
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
def _matmul_128x128x32_cg(
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


def custom_kernel(data: input_t) -> output_t:
    a, b, c = data
    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[1]
    shape = (m, n, k)

    if shape == (2048, 3072, 2048):
        grid = ((m // 128) * (n // 128),)
        _matmul_128x128x32_default[grid](
            a,
            b,
            c,
            M=m,
            N=n,
            K=k,
            GROUP_M=8,
            num_warps=4,
            num_stages=4,
        )
        return c

    if shape == (4096, 5120, 4096):
        grid = ((m // 128) * (n // 128),)
        _matmul_128x128x32_cg[grid](
            a,
            b,
            c,
            M=m,
            N=n,
            K=k,
            GROUP_M=16,
            num_warps=4,
            num_stages=4,
        )
        return c

    torch.mm(a, b, out=c)
    return c
