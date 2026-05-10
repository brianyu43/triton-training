from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def add_base(a_ptr, b_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK_SIZE), BLOCK_SIZE)
    a = tl.load(a_ptr + offs)
    b = tl.load(b_ptr + offs)
    tl.store(out_ptr + offs, a + b)


@triton.jit
def add_no_hint(a_ptr, b_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    a = tl.load(a_ptr + offs)
    b = tl.load(b_ptr + offs)
    tl.store(out_ptr + offs, a + b)


@triton.jit
def add_evict_first(a_ptr, b_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK_SIZE), BLOCK_SIZE)
    a = tl.load(a_ptr + offs, eviction_policy="evict_first")
    b = tl.load(b_ptr + offs, eviction_policy="evict_first")
    tl.store(out_ptr + offs, a + b)


@triton.jit
def add_cache_cg(a_ptr, b_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK_SIZE), BLOCK_SIZE)
    a = tl.load(a_ptr + offs, cache_modifier=".cg", eviction_policy="evict_first")
    b = tl.load(b_ptr + offs, cache_modifier=".cg", eviction_policy="evict_first")
    tl.store(out_ptr + offs, a + b)


@triton.jit
def add_store_cg(a_ptr, b_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK_SIZE), BLOCK_SIZE)
    a = tl.load(a_ptr + offs, eviction_policy="evict_first")
    b = tl.load(b_ptr + offs, eviction_policy="evict_first")
    tl.store(out_ptr + offs, a + b, cache_modifier=".cg")


def make_inputs(size: int):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(size)
    a = torch.randn(size, size, device="cuda", dtype=torch.float16, generator=gen).contiguous()
    b = torch.randn(size, size, device="cuda", dtype=torch.float16, generator=gen).contiguous()
    out = torch.empty(size, size, device="cuda", dtype=torch.float16).contiguous()
    return a, b, out


def time_ms(fn, warmup=20, iters=120):
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
    return 3 * size * size * 2 / (ms * 1e-3) / 1e9


def main() -> None:
    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    print("torch", torch.__version__, "triton", triton.__version__)
    kernels = [
        ("base", add_base),
        ("no_hint", add_no_hint),
        ("evict_first", add_evict_first),
        ("cache_cg", add_cache_cg),
        ("store_cg", add_store_cg),
    ]
    sizes = [4096, 8192, 16384]
    blocks = [128, 256, 512, 1024, 2048, 4096]
    warps_by_block = {
        128: [4],
        256: [4, 8],
        512: [4, 8],
        1024: [4, 8],
        2048: [4, 8],
        4096: [4, 8],
    }
    print("kernel,size,block,warps,best_us,gbps")
    for size in sizes:
        a, b, out = make_inputs(size)
        for name, kernel in kernels:
            for block in blocks:
                grid = (triton.cdiv(a.numel(), block),)
                for warps in warps_by_block[block]:
                    try:
                        fn = lambda kernel=kernel, grid=grid, block=block, warps=warps: kernel[
                            grid
                        ](a, b, out, BLOCK_SIZE=block, num_warps=warps)
                        ms = time_ms(fn)
                        print(f"{name},{size},{block},{warps},{ms * 1000:.3f},{gbps(size, ms):.3f}")
                    except Exception as exc:
                        print(f"{name},{size},{block},{warps},ERROR,{type(exc).__name__}:{exc}")


if __name__ == "__main__":
    main()
