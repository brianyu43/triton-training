from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.cpp_extension import load_inline


@dataclass(frozen=True)
class ProbeResult:
    name: str
    mean_us: float
    best_us: float
    worst_us: float
    bytes_moved: int
    ok: bool | None = None
    abs_err: float | None = None

    @property
    def gbps(self) -> float:
        if self.mean_us <= 0:
            return 0.0
        return self.bytes_moved / (self.mean_us * 1e-6) / 1e9


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

void empty_launch(torch::Tensor output);
void zero_launch(torch::Tensor output);
void read_partial_tile4(torch::Tensor x, torch::Tensor partials, int blocks, int threads);
void final_reduce(torch::Tensor partials, torch::Tensor output, int blocks, int threads);
void two_pass_tile4(torch::Tensor x, torch::Tensor output, torch::Tensor partials, int blocks, int threads);
void atomic_tile4(torch::Tensor x, torch::Tensor output, int blocks, int threads);
void atomic_tail_only(torch::Tensor output, int blocks, int threads);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("empty_launch", &empty_launch, "empty launch");
  m.def("zero_launch", &zero_launch, "zero launch");
  m.def("read_partial_tile4", &read_partial_tile4, "read partial tile4");
  m.def("final_reduce", &final_reduce, "final reduce");
  m.def("two_pass_tile4", &two_pass_tile4, "two pass tile4");
  m.def("atomic_tile4", &atomic_tile4, "atomic tile4");
  m.def("atomic_tail_only", &atomic_tail_only, "atomic tail only");
}
"""

    cuda_src = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kTileVecsPerThread = 4;

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

__device__ __forceinline__ float sum4(float4 v) {
  return (v.x + v.y) + (v.z + v.w);
}

__global__ void empty_kernel(float* __restrict__ output) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    output[0] = output[0];
  }
}

__global__ void zero_kernel(float* __restrict__ output) {
  output[0] = 0.0f;
}

__global__ void read_partial_tile4_kernel(
    const float* __restrict__ x,
    float* __restrict__ partials,
    int64_t n) {
  const int64_t n4 = n >> 2;
  const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
  const int64_t tile = static_cast<int64_t>(blockDim.x) * kTileVecsPerThread;
  const int64_t grid_tile = static_cast<int64_t>(gridDim.x) * tile;
  int64_t base = static_cast<int64_t>(blockIdx.x) * tile + threadIdx.x;

  float s0 = 0.0f;
  float s1 = 0.0f;
  float s2 = 0.0f;
  float s3 = 0.0f;

  for (; base < n4; base += grid_tile) {
    const int64_t i0 = base;
    const int64_t i1 = base + blockDim.x;
    const int64_t i2 = base + 2 * blockDim.x;
    const int64_t i3 = base + 3 * blockDim.x;
    if (i0 < n4) s0 += sum4(x4[i0]);
    if (i1 < n4) s1 += sum4(x4[i1]);
    if (i2 < n4) s2 += sum4(x4[i2]);
    if (i3 < n4) s3 += sum4(x4[i3]);
  }

  float sum = (s0 + s1) + (s2 + s3);
  const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
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

__global__ void final_reduce_kernel(
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

__global__ void atomic_tile4_kernel(
    const float* __restrict__ x,
    float* __restrict__ output,
    int64_t n) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    output[0] = 0.0f;
  }

  const int64_t n4 = n >> 2;
  const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
  const int64_t tile = static_cast<int64_t>(blockDim.x) * kTileVecsPerThread;
  const int64_t grid_tile = static_cast<int64_t>(gridDim.x) * tile;
  int64_t base = static_cast<int64_t>(blockIdx.x) * tile + threadIdx.x;

  float s0 = 0.0f;
  float s1 = 0.0f;
  float s2 = 0.0f;
  float s3 = 0.0f;

  for (; base < n4; base += grid_tile) {
    const int64_t i0 = base;
    const int64_t i1 = base + blockDim.x;
    const int64_t i2 = base + 2 * blockDim.x;
    const int64_t i3 = base + 3 * blockDim.x;
    if (i0 < n4) s0 += sum4(x4[i0]);
    if (i1 < n4) s1 += sum4(x4[i1]);
    if (i2 < n4) s2 += sum4(x4[i2]);
    if (i3 < n4) s3 += sum4(x4[i3]);
  }

  float sum = (s0 + s1) + (s2 + s3);
  const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
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

__global__ void atomic_tail_only_kernel(float* __restrict__ output) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    output[0] = 0.0f;
  }
  if (threadIdx.x == 0) {
    atomicAdd(output, 1.0f);
  }
}

}  // namespace

void empty_launch(torch::Tensor output) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  empty_kernel<<<1, 1, 0, stream>>>(output.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void zero_launch(torch::Tensor output) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  zero_kernel<<<1, 1, 0, stream>>>(output.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void read_partial_tile4(torch::Tensor x, torch::Tensor partials, int blocks, int threads) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  read_partial_tile4_kernel<<<blocks, threads, 0, stream>>>(
      x.data_ptr<float>(), partials.data_ptr<float>(), x.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void final_reduce(torch::Tensor partials, torch::Tensor output, int blocks, int threads) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  final_reduce_kernel<<<1, threads, 0, stream>>>(
      partials.data_ptr<float>(), output.data_ptr<float>(), blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void two_pass_tile4(torch::Tensor x, torch::Tensor output, torch::Tensor partials, int blocks, int threads) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  read_partial_tile4_kernel<<<blocks, threads, 0, stream>>>(
      x.data_ptr<float>(), partials.data_ptr<float>(), x.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  final_reduce_kernel<<<1, threads, 0, stream>>>(
      partials.data_ptr<float>(), output.data_ptr<float>(), blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void atomic_tile4(torch::Tensor x, torch::Tensor output, int blocks, int threads) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  atomic_tile4_kernel<<<blocks, threads, 0, stream>>>(
      x.data_ptr<float>(), output.data_ptr<float>(), x.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void atomic_tail_only(torch::Tensor output, int blocks, int threads) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  atomic_tail_only_kernel<<<blocks, threads, 0, stream>>>(output.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

    return load_inline(
        name="vectorsum_v2_a100_roofline_probe_ext",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


def timed(
    name: str,
    fn: Callable[[], torch.Tensor | None],
    bytes_moved: int,
    reps: int,
    warmups: int,
) -> ProbeResult:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()

    durations = []
    for _ in range(reps):
        clear_l2_cache()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        durations.append(start.elapsed_time(end) * 1000.0)

    return ProbeResult(
        name=name,
        mean_us=sum(durations) / len(durations),
        best_us=min(durations),
        worst_us=max(durations),
        bytes_moved=bytes_moved,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=52_428_800)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--blocks-per-sm", type=int, default=12)
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="Optional probe names to run. Defaults to all probes.",
    )
    args = parser.parse_args()

    torch.cuda.init()
    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    blocks = sms * args.blocks_per_sm
    input_bytes = args.size * 4

    print(
        f"gpu={props.name} sms={sms} size={args.size} seed={args.seed} "
        f"threads={args.threads} blocks_per_sm={args.blocks_per_sm} blocks={blocks} reps={args.reps}",
        flush=True,
    )

    x = generate_input(args.size, args.seed)
    dst = torch.empty_like(x)
    output = torch.empty(1, device="cuda", dtype=torch.float32)
    partials = torch.empty(blocks, device="cuda", dtype=torch.float32)
    expected = x.to(torch.float64).sum().to(torch.float32)
    ext = build_ext()

    probes: list[tuple[str, Callable[[], torch.Tensor | None], int]] = [
        ("empty_launch", lambda: ext.empty_launch(output), 0),
        ("zero_launch", lambda: ext.zero_launch(output), 4),
        ("torch_copy_d2d", lambda: dst.copy_(x), input_bytes * 2),
        ("torch_sum_fp32", lambda: x.sum(), input_bytes),
        (
            "read_partial_tile4",
            lambda: ext.read_partial_tile4(x, partials, blocks, args.threads),
            input_bytes + blocks * 4,
        ),
        (
            "final_reduce_only",
            lambda: ext.final_reduce(partials, output, blocks, args.threads),
            blocks * 4 + 4,
        ),
        (
            "two_pass_tile4",
            lambda: ext.two_pass_tile4(x, output, partials, blocks, args.threads),
            input_bytes + blocks * 8 + 4,
        ),
        (
            "atomic_tail_only",
            lambda: ext.atomic_tail_only(output, blocks, args.threads),
            blocks * 4 + 4,
        ),
        ("v11_atomic_tile4", lambda: ext.atomic_tile4(x, output, blocks, args.threads), input_bytes + 4),
    ]

    if args.only:
        selected = set(args.only)
        probes = [probe for probe in probes if probe[0] in selected]
        missing = selected - {probe[0] for probe in probes}
        if missing:
            valid = ", ".join(
                [
                    "empty_launch",
                    "zero_launch",
                    "torch_copy_d2d",
                    "torch_sum_fp32",
                    "read_partial_tile4",
                    "final_reduce_only",
                    "two_pass_tile4",
                    "atomic_tail_only",
                    "v11_atomic_tile4",
                ]
            )
            raise ValueError(f"unknown probe(s): {sorted(missing)}. valid probes: {valid}")

    if not args.only:
        ext.read_partial_tile4(x, partials, blocks, args.threads)
        ext.final_reduce(partials, output, blocks, args.threads)
        torch.cuda.synchronize()

    for name, fn, bytes_moved in probes:
        result = timed(name, fn, bytes_moved, args.reps, args.warmups)
        ok_text = ""
        if name in {"two_pass_tile4", "v11_atomic_tile4"}:
            abs_err = float(torch.abs(output[0] - expected).item())
            ok = bool(torch.allclose(output[0], expected, rtol=1e-5, atol=1e-8))
            ok_text = f" ok={ok} abs_err={abs_err:.6g}"
        print(
            f"{result.name:18s} mean_us={result.mean_us:9.3f} "
            f"best_us={result.best_us:9.3f} worst_us={result.worst_us:9.3f} "
            f"bytes={result.bytes_moved:12d} gbps={result.gbps:8.1f}{ok_text}",
            flush=True,
        )


if __name__ == "__main__":
    main()
