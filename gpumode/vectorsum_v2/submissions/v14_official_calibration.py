#!POPCORN leaderboard vectorsum_v2
#!POPCORN gpus A100

from __future__ import annotations

import os
from typing import Any

import torch
from torch.utils.cpp_extension import load_inline

from task import input_t, output_t

_EXT: Any = None


def _get_ext() -> Any:
    global _EXT
    if _EXT is not None:
        return _EXT

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    os.environ["MAX_JOBS"] = "4"

    cpp_src = r"""
#include <torch/extension.h>

torch::Tensor vectorsum_forward(torch::Tensor x, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vectorsum_forward", &vectorsum_forward, "A100 vectorsum official calibration forward");
}
"""

    cuda_src = r"""
#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kBlockSize = 256;
constexpr int kBlocksPerSm = 12;

int g_blocks = 0;

__device__ __forceinline__ float warp_sum(float value) {
  value += __shfl_down_sync(0xffffffff, value, 16);
  value += __shfl_down_sync(0xffffffff, value, 8);
  value += __shfl_down_sync(0xffffffff, value, 4);
  value += __shfl_down_sync(0xffffffff, value, 2);
  value += __shfl_down_sync(0xffffffff, value, 1);
  return value;
}

__device__ __forceinline__ float block_sum(float value) {
  __shared__ float warp_partials[8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  value = warp_sum(value);
  if (lane == 0) {
    warp_partials[warp] = value;
  }
  __syncthreads();

  value = (threadIdx.x < 8) ? warp_partials[lane] : 0.0f;
  if (warp == 0) {
    value = warp_sum(value);
  }
  return value;
}

__device__ __forceinline__ float sum4(float4 v) {
  return (v.x + v.y) + (v.z + v.w);
}

__global__ __launch_bounds__(256, 8) void atomic_sum_kernel(
    const float4* __restrict__ x4,
    float* __restrict__ output,
    int n4) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    output[0] = 0.0f;
  }

  const int stride = blockDim.x * gridDim.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;

  for (; idx < n4; idx += stride) {
    sum += sum4(x4[idx]);
  }

  sum = block_sum(sum);
  if (threadIdx.x == 0) {
    atomicAdd(output, sum);
  }
}

__global__ void small_sum_kernel(
    const float* __restrict__ x,
    float* __restrict__ output,
    int n) {
  float sum = 0.0f;
  for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
    sum += x[idx];
  }
  sum = block_sum(sum);
  if (threadIdx.x == 0) {
    output[0] = sum;
  }
}

void ensure_state() {
  if (g_blocks > 0) {
    return;
  }
  int device = 0;
  cudaDeviceProp prop;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  g_blocks = prop.multiProcessorCount * kBlocksPerSm;
}

}  // namespace

torch::Tensor vectorsum_forward(torch::Tensor x, torch::Tensor output) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(output.is_cuda(), "output must be CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(output.scalar_type() == torch::kFloat32, "output must be float32");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(output.numel() >= 1, "output must have at least one element");

  const int64_t n_i64 = x.numel();
  TORCH_CHECK(n_i64 <= static_cast<int64_t>(2147483647), "n must fit int32");

  if (n_i64 <= 4096 || (n_i64 & 3)) {
    small_sum_kernel<<<1, kBlockSize>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), static_cast<int>(n_i64));
  } else {
    const int64_t n4_i64 = n_i64 >> 2;
    ensure_state();
    atomic_sum_kernel<<<g_blocks, kBlockSize>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        output.data_ptr<float>(),
        static_cast<int>(n4_i64));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output.index({0});
}
"""

    _EXT = load_inline(
        name="vectorsum_v2_a100_v14_official_calibration_ext",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return _EXT


def custom_kernel(data: input_t) -> output_t:
    values, output = data
    return _get_ext().vectorsum_forward(values, output)
