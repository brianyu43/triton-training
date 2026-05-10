from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import torch
from torch.utils.cpp_extension import load_inline


@dataclass(frozen=True)
class Result:
    kind: str
    threads: int
    blocks_per_sm: int
    blocks: int
    mean_us: float
    best_us: float
    worst_us: float
    ok: bool
    abs_err: float


def clear_l2_cache() -> None:
    dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda")
    dummy.fill_(42)
    del dummy


def generate_input(size: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    data = torch.randn(size, device="cuda", dtype=torch.float32, generator=gen).contiguous()

    offset_gen = torch.Generator(device="cuda")
    offset_gen.manual_seed(seed + 1)
    scale_gen = torch.Generator(device="cuda")
    scale_gen.manual_seed(seed + 2)

    offset = (torch.rand(1, device="cuda", generator=offset_gen) * 200 - 100).item()
    scale = (torch.rand(1, device="cuda", generator=scale_gen) * 9.9 + 0.1).item()
    return (data * scale + offset).contiguous()


def build_ext():
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    os.environ["MAX_JOBS"] = "4"

    cpp_src = r"""
#include <torch/extension.h>

void atomic_vec4(torch::Tensor x, torch::Tensor output, int blocks, int threads);
void two_pass_vec4(torch::Tensor x, torch::Tensor output, torch::Tensor partials, int blocks, int threads);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("atomic_vec4", &atomic_vec4, "sweep atomic vec4");
  m.def("two_pass_vec4", &two_pass_vec4, "sweep two pass vec4");
}
"""

    cuda_src = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ float warp_sum(float value) {
  value += __shfl_down_sync(0xffffffff, value, 16);
  value += __shfl_down_sync(0xffffffff, value, 8);
  value += __shfl_down_sync(0xffffffff, value, 4);
  value += __shfl_down_sync(0xffffffff, value, 2);
  value += __shfl_down_sync(0xffffffff, value, 1);
  return value;
}

__device__ __forceinline__ float block_sum_dynamic(float value) {
  __shared__ float warp_partials[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warp_count = blockDim.x >> 5;

  value = warp_sum(value);
  if (lane == 0) {
    warp_partials[warp] = value;
  }
  __syncthreads();

  value = (threadIdx.x < warp_count) ? warp_partials[lane] : 0.0f;
  if (warp == 0) {
    value = warp_sum(value);
  }
  return value;
}

__global__ void atomic_vec4_kernel(
    const float* __restrict__ x,
    float* __restrict__ output,
    int64_t n) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    output[0] = 0.0f;
  }

  float sum = 0.0f;
  const int64_t n4 = n >> 2;
  const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
  const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
  int64_t idx = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x)
      + static_cast<int64_t>(threadIdx.x);

  for (; idx < n4; idx += stride) {
    const float4 v = x4[idx];
    sum += v.x + v.y + v.z + v.w;
  }

  for (int64_t tail = (n4 << 2) + static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       tail < n;
       tail += stride) {
    sum += x[tail];
  }

  sum = block_sum_dynamic(sum);
  if (threadIdx.x == 0) {
    atomicAdd(output, sum);
  }
}

__global__ void partial_vec4_kernel(
    const float* __restrict__ x,
    float* __restrict__ partials,
    int64_t n) {
  float sum = 0.0f;
  const int64_t n4 = n >> 2;
  const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
  const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
  int64_t idx = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x)
      + static_cast<int64_t>(threadIdx.x);

  for (; idx < n4; idx += stride) {
    const float4 v = x4[idx];
    sum += v.x + v.y + v.z + v.w;
  }

  for (int64_t tail = (n4 << 2) + static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       tail < n;
       tail += stride) {
    sum += x[tail];
  }

  sum = block_sum_dynamic(sum);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = sum;
  }
}

__global__ void final_kernel(
    const float* __restrict__ partials,
    float* __restrict__ output,
    int n_partials) {
  float sum = 0.0f;
  for (int idx = threadIdx.x; idx < n_partials; idx += blockDim.x) {
    sum += partials[idx];
  }
  sum = block_sum_dynamic(sum);
  if (threadIdx.x == 0) {
    output[0] = sum;
  }
}

}  // namespace

void atomic_vec4(torch::Tensor x, torch::Tensor output, int blocks, int threads) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  atomic_vec4_kernel<<<blocks, threads, 0, stream>>>(
      x.data_ptr<float>(), output.data_ptr<float>(), x.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void two_pass_vec4(torch::Tensor x, torch::Tensor output, torch::Tensor partials, int blocks, int threads) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  partial_vec4_kernel<<<blocks, threads, 0, stream>>>(
      x.data_ptr<float>(), partials.data_ptr<float>(), x.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  final_kernel<<<1, threads, 0, stream>>>(
      partials.data_ptr<float>(), output.data_ptr<float>(), blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

    return load_inline(
        name="vectorsum_v2_a100_sweep_ext",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


def time_variant(ext, x, expected, kind: str, threads: int, blocks_per_sm: int, sms: int, reps: int) -> Result:
    blocks = sms * blocks_per_sm
    output = torch.empty(1, device="cuda", dtype=torch.float32)
    partials = torch.empty(blocks, device="cuda", dtype=torch.float32)

    if kind == "atomic":
        ext.atomic_vec4(x, output, blocks, threads)
    else:
        ext.two_pass_vec4(x, output, partials, blocks, threads)
    torch.cuda.synchronize()

    abs_err = float(torch.abs(output[0] - expected).item())
    ok = bool(torch.allclose(output[0], expected, rtol=1e-5, atol=1e-8))

    durations = []
    for _ in range(reps):
        clear_l2_cache()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if kind == "atomic":
            ext.atomic_vec4(x, output, blocks, threads)
        else:
            ext.two_pass_vec4(x, output, partials, blocks, threads)
        end.record()
        torch.cuda.synchronize()
        durations.append(start.elapsed_time(end) * 1000.0)

    return Result(
        kind=kind,
        threads=threads,
        blocks_per_sm=blocks_per_sm,
        blocks=blocks,
        mean_us=sum(durations) / len(durations),
        best_us=min(durations),
        worst_us=max(durations),
        ok=ok,
        abs_err=abs_err,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=52_428_800)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--threads", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--blocks-per-sm", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16])
    args = parser.parse_args()

    torch.cuda.init()
    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    print(f"gpu={props.name} sms={sms} size={args.size} seed={args.seed} reps={args.reps}")

    x = generate_input(args.size, args.seed)
    expected = x.to(torch.float64).sum().to(torch.float32)
    ext = build_ext()

    results: list[Result] = []
    for kind in ("atomic", "two_pass"):
      for threads in args.threads:
        for bps in args.blocks_per_sm:
          result = time_variant(ext, x, expected, kind, threads, bps, sms, args.reps)
          results.append(result)
          print(
              f"{result.kind:8s} threads={result.threads:4d} bps={result.blocks_per_sm:2d} "
              f"blocks={result.blocks:4d} mean_us={result.mean_us:9.3f} "
              f"best_us={result.best_us:9.3f} worst_us={result.worst_us:9.3f} "
              f"ok={result.ok} abs_err={result.abs_err:.6g}",
              flush=True,
          )

    best = min((r for r in results if r.ok), key=lambda r: r.mean_us)
    print(
        f"BEST kind={best.kind} threads={best.threads} bps={best.blocks_per_sm} "
        f"blocks={best.blocks} mean_us={best.mean_us:.3f} best_us={best.best_us:.3f}"
    )


if __name__ == "__main__":
    main()
