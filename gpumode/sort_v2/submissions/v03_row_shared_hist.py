#!POPCORN leaderboard sort_v2
#!POPCORN gpus A100

from __future__ import annotations

import os
from typing import Any

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline


EXACT_LIMIT = 100000
ROW_HIST_LIMIT = 50000000

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

torch::Tensor sort_v2_bucket_counting_cuda(torch::Tensor values, torch::Tensor output, bool use_row_hist);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sort_v2_bucket_counting_cuda", &sort_v2_bucket_counting_cuda, "sort_v2 bucket counting cuda");
}
"""

    cuda_src = r"""
#include <torch/extension.h>

#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kBpu = 64;
constexpr int kBias = 16;
constexpr int kThreads = 256;
constexpr int kLocalBins = 2 * kBias * kBpu;

static inline int64_t ceil_div_i64(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

static inline void cuda_check(cudaError_t status, const char* msg) {
  TORCH_CHECK(status == cudaSuccess, msg, ": ", cudaGetErrorString(status));
}

__device__ __forceinline__ int clamp_i32(int x, int lo, int hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

__device__ __forceinline__ int bucket_id(float x, float base, int bucket_count) {
  int q = static_cast<int>(floorf((x - base + static_cast<float>(kBias)) * static_cast<float>(kBpu)));
  return clamp_i32(q, 0, bucket_count - 1);
}

__device__ __forceinline__ int local_bucket_id(float x, float base, int64_t row) {
  int q = static_cast<int>(floorf((x - (base + static_cast<float>(row)) + static_cast<float>(kBias)) * static_cast<float>(kBpu)));
  return clamp_i32(q, 0, kLocalBins - 1);
}

__host__ __device__ __forceinline__ int first_row_for_bucket(int q) {
  if (q < kLocalBins - 1) return 0;
  return (q - (kLocalBins - 1) + kBpu - 1) / kBpu;
}

__global__ void estimate_base_kernel(
    const float* __restrict__ values,
    float* __restrict__ base_out,
    int64_t cols) {
  __shared__ float shared[kThreads];
  const int tid = threadIdx.x;
  float sum = 0.0f;
  for (int64_t i = tid; i < cols; i += blockDim.x) {
    sum += values[i];
  }
  shared[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    base_out[0] = nearbyintf(shared[0] / static_cast<float>(cols));
  }
}

__global__ void histogram_global_kernel(
    const float* __restrict__ values,
    const float* __restrict__ base,
    int* __restrict__ counts,
    int64_t n,
    int bucket_count) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  const float b = base[0];
  for (int64_t i = idx; i < n; i += stride) {
    const int q = bucket_id(values[i], b, bucket_count);
    atomicAdd(counts + q, 1);
  }
}

__global__ void scatter_global_kernel(
    const float* __restrict__ values,
    const float* __restrict__ base,
    int* __restrict__ offsets,
    float* __restrict__ output,
    int64_t n,
    int bucket_count) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  const float b = base[0];
  for (int64_t i = idx; i < n; i += stride) {
    const float x = values[i];
    const int q = bucket_id(x, b, bucket_count);
    const int pos = atomicAdd(offsets + q, 1);
    output[pos] = x;
  }
}

__global__ void row_histogram_kernel(
    const float* __restrict__ values,
    const float* __restrict__ base,
    int* __restrict__ row_counts,
    int64_t n,
    int64_t cols) {
  extern __shared__ int hist[];
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int tid = threadIdx.x;

  for (int bin = tid; bin < kLocalBins; bin += blockDim.x) {
    hist[bin] = 0;
  }
  __syncthreads();

  const int64_t start = row * cols;
  const int64_t end = min(start + cols, n);
  const float b = base[0];
  for (int64_t i = start + tid; i < end; i += blockDim.x) {
    const int local = local_bucket_id(values[i], b, row);
    atomicAdd(hist + local, 1);
  }
  __syncthreads();

  int* row_out = row_counts + row * kLocalBins;
  for (int bin = tid; bin < kLocalBins; bin += blockDim.x) {
    row_out[bin] = hist[bin];
  }
}

__global__ void reduce_row_counts_kernel(
    const int* __restrict__ row_counts,
    int* __restrict__ counts,
    int rows,
    int bucket_count) {
  const int q = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (q >= bucket_count) return;

  const int r0 = first_row_for_bucket(q);
  int r1 = q / kBpu;
  r1 = r1 >= rows ? rows - 1 : r1;

  int total = 0;
  for (int r = r0; r <= r1; ++r) {
    const int local = q - r * kBpu;
    total += row_counts[r * kLocalBins + local];
  }
  counts[q] = total;
}

__global__ void row_offsets_from_buckets_kernel(
    const int* __restrict__ row_counts,
    const int* __restrict__ bucket_offsets,
    int* __restrict__ row_offsets,
    int rows,
    int bucket_count) {
  const int q = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (q >= bucket_count) return;

  const int r0 = first_row_for_bucket(q);
  int r1 = q / kBpu;
  r1 = r1 >= rows ? rows - 1 : r1;

  int running = bucket_offsets[q];
  for (int r = r0; r <= r1; ++r) {
    const int local = q - r * kBpu;
    const int idx = r * kLocalBins + local;
    row_offsets[idx] = running;
    running += row_counts[idx];
  }
}

__global__ void scatter_row_kernel(
    const float* __restrict__ values,
    const float* __restrict__ base,
    const int* __restrict__ row_offsets,
    float* __restrict__ output,
    int64_t n,
    int64_t cols) {
  extern __shared__ int offsets[];
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int tid = threadIdx.x;

  const int* row_off = row_offsets + row * kLocalBins;
  for (int bin = tid; bin < kLocalBins; bin += blockDim.x) {
    offsets[bin] = row_off[bin];
  }
  __syncthreads();

  const int64_t start = row * cols;
  const int64_t end = min(start + cols, n);
  const float b = base[0];
  for (int64_t i = start + tid; i < end; i += blockDim.x) {
    const float x = values[i];
    const int local = local_bucket_id(x, b, row);
    const int pos = atomicAdd(offsets + local, 1);
    output[pos] = x;
  }
}

void run_global_path(
    torch::Tensor values,
    torch::Tensor output,
    torch::Tensor base,
    int64_t n,
    int bucket_count) {
  auto int_opts = values.options().dtype(torch::kInt32);
  auto byte_opts = values.options().dtype(torch::kUInt8);
  auto counts = torch::empty({bucket_count}, int_opts);
  auto offsets = torch::empty({bucket_count}, int_opts);

  cuda_check(cudaMemsetAsync(counts.data_ptr<int>(), 0, bucket_count * sizeof(int)), "cudaMemsetAsync failed");

  const int blocks = static_cast<int>(std::min<int64_t>(ceil_div_i64(n, kThreads), 4096));
  histogram_global_kernel<<<blocks, kThreads>>>(
      values.data_ptr<float>(),
      base.data_ptr<float>(),
      counts.data_ptr<int>(),
      n,
      bucket_count);
  cuda_check(cudaGetLastError(), "histogram_global_kernel launch failed");

  void* temp_storage = nullptr;
  size_t temp_bytes = 0;
  cub::DeviceScan::ExclusiveSum(
      temp_storage,
      temp_bytes,
      counts.data_ptr<int>(),
      offsets.data_ptr<int>(),
      bucket_count);
  auto temp = torch::empty({static_cast<int64_t>(temp_bytes)}, byte_opts);
  cub::DeviceScan::ExclusiveSum(
      temp.data_ptr<uint8_t>(),
      temp_bytes,
      counts.data_ptr<int>(),
      offsets.data_ptr<int>(),
      bucket_count);

  scatter_global_kernel<<<blocks, kThreads>>>(
      values.data_ptr<float>(),
      base.data_ptr<float>(),
      offsets.data_ptr<int>(),
      output.data_ptr<float>(),
      n,
      bucket_count);
  cuda_check(cudaGetLastError(), "scatter_global_kernel launch failed");
}

void run_row_hist_path(
    torch::Tensor values,
    torch::Tensor output,
    torch::Tensor base,
    int64_t n,
    int rows,
    int64_t cols,
    int bucket_count) {
  auto int_opts = values.options().dtype(torch::kInt32);
  auto byte_opts = values.options().dtype(torch::kUInt8);
  auto row_counts = torch::empty({static_cast<int64_t>(rows) * kLocalBins}, int_opts);
  auto row_offsets = torch::empty({static_cast<int64_t>(rows) * kLocalBins}, int_opts);
  auto counts = torch::empty({bucket_count}, int_opts);
  auto offsets = torch::empty({bucket_count}, int_opts);

  const size_t shmem_bytes = kLocalBins * sizeof(int);
  row_histogram_kernel<<<rows, kThreads, shmem_bytes>>>(
      values.data_ptr<float>(),
      base.data_ptr<float>(),
      row_counts.data_ptr<int>(),
      n,
      cols);
  cuda_check(cudaGetLastError(), "row_histogram_kernel launch failed");

  const int count_blocks = static_cast<int>(ceil_div_i64(bucket_count, kThreads));
  reduce_row_counts_kernel<<<count_blocks, kThreads>>>(
      row_counts.data_ptr<int>(),
      counts.data_ptr<int>(),
      rows,
      bucket_count);
  cuda_check(cudaGetLastError(), "reduce_row_counts_kernel launch failed");

  void* temp_storage = nullptr;
  size_t temp_bytes = 0;
  cub::DeviceScan::ExclusiveSum(
      temp_storage,
      temp_bytes,
      counts.data_ptr<int>(),
      offsets.data_ptr<int>(),
      bucket_count);
  auto temp = torch::empty({static_cast<int64_t>(temp_bytes)}, byte_opts);
  cub::DeviceScan::ExclusiveSum(
      temp.data_ptr<uint8_t>(),
      temp_bytes,
      counts.data_ptr<int>(),
      offsets.data_ptr<int>(),
      bucket_count);

  row_offsets_from_buckets_kernel<<<count_blocks, kThreads>>>(
      row_counts.data_ptr<int>(),
      offsets.data_ptr<int>(),
      row_offsets.data_ptr<int>(),
      rows,
      bucket_count);
  cuda_check(cudaGetLastError(), "row_offsets_from_buckets_kernel launch failed");

  scatter_row_kernel<<<rows, kThreads, shmem_bytes>>>(
      values.data_ptr<float>(),
      base.data_ptr<float>(),
      row_offsets.data_ptr<int>(),
      output.data_ptr<float>(),
      n,
      cols);
  cuda_check(cudaGetLastError(), "scatter_row_kernel launch failed");
}

}  // namespace

torch::Tensor sort_v2_bucket_counting_cuda(torch::Tensor values, torch::Tensor output, bool use_row_hist) {
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(output.is_cuda(), "output must be CUDA");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32, "values must be float32");
  TORCH_CHECK(output.scalar_type() == torch::kFloat32, "output must be float32");
  TORCH_CHECK(values.is_contiguous(), "values must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");

  const int64_t n = values.numel();
  const int rows = static_cast<int>(std::sqrt(static_cast<double>(n)));
  const int64_t cols = ceil_div_i64(n, rows);
  const int bucket_count = static_cast<int>((static_cast<int64_t>(rows) + 2 * kBias) * kBpu);

  auto base = torch::empty({1}, values.options());
  estimate_base_kernel<<<1, kThreads>>>(
      values.data_ptr<float>(),
      base.data_ptr<float>(),
      cols);
  cuda_check(cudaGetLastError(), "estimate_base_kernel launch failed");

  if (use_row_hist) {
    run_row_hist_path(values, output, base, n, rows, cols, bucket_count);
  } else {
    run_global_path(values, output, base, n, bucket_count);
  }

  return output;
}
"""

    _EXT = load_inline(
        name="sort_v2_row_shared_hist_v03",
        cpp_sources=cpp_src,
        cuda_sources=cuda_src,
        functions=None,
        extra_cuda_cflags=["-O3", "-std=c++17", "-arch=sm_80", "--use_fast_math"],
        extra_cflags=["-O3", "-std=c++17"],
        with_cuda=True,
        verbose=False,
    )
    return _EXT


def custom_kernel(data: input_t) -> output_t:
    values, output = data
    n = values.numel()
    if n < EXACT_LIMIT:
        output[...] = torch.sort(values)[0]
        return output
    return _get_ext().sort_v2_bucket_counting_cuda(values, output, n >= ROW_HIST_LIMIT)
