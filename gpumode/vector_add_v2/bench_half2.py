from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def add_half2_i32(a_ptr, b_ptr, out_ptr, BLOCK_PAIRS: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_PAIRS + tl.arange(0, BLOCK_PAIRS)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK_PAIRS), BLOCK_PAIRS)

    a_i32 = a_ptr.to(tl.pointer_type(tl.int32))
    b_i32 = b_ptr.to(tl.pointer_type(tl.int32))
    out_i32 = out_ptr.to(tl.pointer_type(tl.int32))

    a = tl.load(a_i32 + offs, eviction_policy="evict_first")
    b = tl.load(b_i32 + offs, eviction_policy="evict_first")
    c = tl.inline_asm_elementwise(
        asm="add.rn.f16x2 $0, $1, $2;",
        constraints="=r,r,r",
        args=[a, b],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    tl.store(out_i32 + offs, c)


@triton.jit
def add_half2_i32_store_cg(a_ptr, b_ptr, out_ptr, BLOCK_PAIRS: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_PAIRS + tl.arange(0, BLOCK_PAIRS)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK_PAIRS), BLOCK_PAIRS)

    a_i32 = a_ptr.to(tl.pointer_type(tl.int32))
    b_i32 = b_ptr.to(tl.pointer_type(tl.int32))
    out_i32 = out_ptr.to(tl.pointer_type(tl.int32))

    a = tl.load(a_i32 + offs, eviction_policy="evict_first")
    b = tl.load(b_i32 + offs, eviction_policy="evict_first")
    c = tl.inline_asm_elementwise(
        asm="add.rn.f16x2 $0, $1, $2;",
        constraints="=r,r,r",
        args=[a, b],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    tl.store(out_i32 + offs, c, cache_modifier=".cg")


@triton.jit
def add_half2_i32_store_cs(a_ptr, b_ptr, out_ptr, BLOCK_PAIRS: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_PAIRS + tl.arange(0, BLOCK_PAIRS)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK_PAIRS), BLOCK_PAIRS)

    a_i32 = a_ptr.to(tl.pointer_type(tl.int32))
    b_i32 = b_ptr.to(tl.pointer_type(tl.int32))
    out_i32 = out_ptr.to(tl.pointer_type(tl.int32))

    a = tl.load(a_i32 + offs, eviction_policy="evict_first")
    b = tl.load(b_i32 + offs, eviction_policy="evict_first")
    c = tl.inline_asm_elementwise(
        asm="add.rn.f16x2 $0, $1, $2;",
        constraints="=r,r,r",
        args=[a, b],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    tl.store(out_i32 + offs, c, cache_modifier=".cs")


def make_inputs(size: int, seed: int):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    a = torch.randn(size, size, device="cuda", dtype=torch.float16, generator=gen).contiguous()
    b = torch.randn(size, size, device="cuda", dtype=torch.float16, generator=gen).contiguous()
    out = torch.empty(size, size, device="cuda", dtype=torch.float16).contiguous()
    return a, b, out


def time_ms(fn, warmup=30, iters=200):
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
        ("half2_i32", add_half2_i32),
        ("half2_i32_store_cg", add_half2_i32_store_cg),
        ("half2_i32_store_cs", add_half2_i32_store_cs),
    ]
    print("kernel,size,block_pairs,warps,best_us,gbps,max_err")
    for size in [4096, 8192, 16384]:
        a, b, out = make_inputs(size, size)
        ref = a + b
        for name, kernel in kernels:
            for block_pairs in [128, 256, 512, 1024, 2048, 4096]:
                grid = (triton.cdiv(a.numel() // 2, block_pairs),)
                for warps in [4, 8]:
                    if block_pairs == 128 and warps == 8:
                        continue
                    fn = lambda kernel=kernel, grid=grid, block_pairs=block_pairs, warps=warps: kernel[
                        grid
                    ](a, b, out, BLOCK_PAIRS=block_pairs, num_warps=warps)
                    try:
                        fn()
                        torch.cuda.synchronize()
                        max_err = (out - ref).abs().max().item()
                        ms = time_ms(fn)
                        print(f"{name},{size},{block_pairs},{warps},{ms * 1000:.3f},{gbps(size, ms):.3f},{max_err:.3e}")
                    except Exception as exc:
                        print(f"{name},{size},{block_pairs},{warps},ERROR,0,{type(exc).__name__}:{exc}")


if __name__ == "__main__":
    main()
