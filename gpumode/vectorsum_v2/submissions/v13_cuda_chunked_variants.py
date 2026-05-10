#!POPCORN leaderboard vectorsum_py
#!POPCORN gpus A100

from __future__ import annotations

import os
import re
from typing import Any

import torch
from torch.utils.cpp_extension import load_inline

from task import input_t, output_t

_EXT: Any = None


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _env_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", value)


def _get_ext() -> Any:
    global _EXT
    if _EXT is not None:
        return _EXT

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    os.environ["MAX_JOBS"] = "4"

    blocks_per_sm = _env_int("V13_BLOCKS_PER_SM", 12)
    zero_mode = os.getenv("V13_ZERO", "inside").strip().lower()
    layout = os.getenv("V13_LAYOUT", "chunked").strip().lower()
    maxrreg = os.getenv("V13_MAXRREG", "").strip()

    if zero_mode not in {"inside", "memset"}:
        raise ValueError("V13_ZERO must be 'inside' or 'memset'")
    if layout not in {"chunked", "grid"}:
        raise ValueError("V13_LAYOUT must be 'chunked' or 'grid'")
    if maxrreg and (not maxrreg.isdigit() or int(maxrreg) <= 0):
        raise ValueError("V13_MAXRREG must be a positive integer")

    cpp_src = r"""
#include <torch/extension.h>

torch::Tensor vectorsum_forward(torch::Tensor x, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vectorsum_forward", &vectorsum_forward, "A100 vectorsum chunked variants forward");
}
"""

    cuda_src = rf"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {{

constexpr int kBlockSize = 256;
constexpr int kBlocksPerSm = {blocks_per_sm};
constexpr bool kUseMemsetZero = {"true" if zero_mode == "memset" else "false"};
constexpr bool kUseChunkedLayout = {"true" if layout == "chunked" else "false"};

int g_blocks = 0;

__device__ __forceinline__ float warp_sum(float value) {{
  value += __shfl_down_sync(0xffffffff, value, 16);
  value += __shfl_down_sync(0xffffffff, value, 8);
  value += __shfl_down_sync(0xffffffff, value, 4);
  value += __shfl_down_sync(0xffffffff, value, 2);
  value += __shfl_down_sync(0xffffffff, value, 1);
  return value;
}}

__device__ __forceinline__ float block_sum(float value) {{
  __shared__ float warp_partials[8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  value = warp_sum(value);
  if (lane == 0) {{
    warp_partials[warp] = value;
  }}
  __syncthreads();

  value = (threadIdx.x < 8) ? warp_partials[lane] : 0.0f;
  if (warp == 0) {{
    value = warp_sum(value);
  }}
  return value;
}}

__device__ __forceinline__ float sum4(float4 v) {{
  return (v.x + v.y) + (v.z + v.w);
}}

template <bool kZeroInside, bool kChunkedLayout>
__global__ __launch_bounds__(256, 8) void atomic_sum_kernel(
    const float4* __restrict__ x4,
    float* __restrict__ output,
    int n4) {{
  if (kZeroInside && blockIdx.x == 0 && threadIdx.x == 0) {{
    output[0] = 0.0f;
  }}

  float sum = 0.0f;
  if (kChunkedLayout) {{
    const int chunk = (n4 + gridDim.x - 1) / gridDim.x;
    const int begin = blockIdx.x * chunk;
    const int end = min(begin + chunk, n4);
    for (int idx = begin + threadIdx.x; idx < end; idx += blockDim.x) {{
      sum += sum4(x4[idx]);
    }}
  }} else {{
    const int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n4; idx += stride) {{
      sum += sum4(x4[idx]);
    }}
  }}

  sum = block_sum(sum);
  if (threadIdx.x == 0) {{
    atomicAdd(output, sum);
  }}
}}

void ensure_state() {{
  if (g_blocks > 0) {{
    return;
  }}
  int device = 0;
  cudaDeviceProp prop;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  g_blocks = prop.multiProcessorCount * kBlocksPerSm;
  if (g_blocks < 1) {{
    g_blocks = 1;
  }}
}}

}}  // namespace

torch::Tensor vectorsum_forward(torch::Tensor x, torch::Tensor output) {{
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(output.is_cuda(), "output must be CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(output.scalar_type() == torch::kFloat32, "output must be float32");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(output.numel() >= 1, "output must have at least one element");
  TORCH_CHECK((x.numel() & 3) == 0, "fast path expects n % 4 == 0");

  const int64_t n4_i64 = x.numel() >> 2;
  TORCH_CHECK(n4_i64 <= static_cast<int64_t>(2147483647), "n/4 must fit int32");
  ensure_state();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (kUseMemsetZero) {{
    C10_CUDA_CHECK(cudaMemsetAsync(output.data_ptr<float>(), 0, sizeof(float), stream));
    atomic_sum_kernel<false, kUseChunkedLayout><<<g_blocks, kBlockSize, 0, stream>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        output.data_ptr<float>(),
        static_cast<int>(n4_i64));
  }} else {{
    atomic_sum_kernel<true, kUseChunkedLayout><<<g_blocks, kBlockSize, 0, stream>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        output.data_ptr<float>(),
        static_cast<int>(n4_i64));
  }}
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output.index({{0}});
}}
"""

    extra_cuda_cflags = ["-O3", "--use_fast_math"]
    if maxrreg:
        extra_cuda_cflags.append(f"-maxrregcount={maxrreg}")

    _EXT = load_inline(
        name=(
            "vectorsum_v2_a100_v13_chunked_"
            f"bps{blocks_per_sm}_{_env_name(layout)}_{_env_name(zero_mode)}_r{maxrreg or 'none'}"
        ),
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        extra_cflags=["-O3"],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False,
    )
    return _EXT


def custom_kernel(data: input_t) -> output_t:
    values, output = data
    n_elements = values.numel()

    if n_elements <= 4096 or (n_elements & 3):
        return values.to(torch.float64).sum().to(torch.float32)

    return _get_ext().vectorsum_forward(values, output)
