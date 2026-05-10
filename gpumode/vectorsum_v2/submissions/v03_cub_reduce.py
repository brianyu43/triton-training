#!POPCORN leaderboard vectorsum_py
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

torch::Tensor vectorsum_cub_forward(torch::Tensor x, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vectorsum_cub_forward", &vectorsum_cub_forward, "A100 vectorsum CUB forward");
}
"""

    cuda_src = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

at::Tensor g_temp_storage;
size_t g_temp_bytes = 0;
int64_t g_cached_n = -1;

void ensure_temp_storage(torch::Tensor x, int64_t n) {
  size_t required_bytes = 0;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cub::DeviceReduce::Sum(
      nullptr,
      required_bytes,
      x.data_ptr<float>(),
      static_cast<float*>(nullptr),
      static_cast<int>(n),
      stream));

  if (!g_temp_storage.defined() || g_temp_bytes < required_bytes || g_cached_n != n) {
    g_temp_storage = torch::empty(
        {static_cast<int64_t>(required_bytes)},
        torch::TensorOptions().device(x.device()).dtype(torch::kUInt8));
    g_temp_bytes = required_bytes;
    g_cached_n = n;
  }
}

}  // namespace

torch::Tensor vectorsum_cub_forward(torch::Tensor x, torch::Tensor output) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(output.is_cuda(), "output must be CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(output.scalar_type() == torch::kFloat32, "output must be float32");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(output.numel() >= 1, "output must have at least one element");

  const int64_t n = x.numel();
  TORCH_CHECK(n <= static_cast<int64_t>(2147483647), "CUB path expects n <= INT_MAX");
  if (n == 0) {
    output.zero_();
    return output.index({0});
  }

  ensure_temp_storage(x, n);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cub::DeviceReduce::Sum(
      g_temp_storage.data_ptr(),
      g_temp_bytes,
      x.data_ptr<float>(),
      output.data_ptr<float>(),
      static_cast<int>(n),
      stream));

  return output.index({0});
}
"""

    _EXT = load_inline(
        name="vectorsum_v2_a100_cub_ext",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return _EXT


def custom_kernel(data: input_t) -> output_t:
    values, output = data
    n_elements = values.numel()

    if n_elements <= 4096:
        return values.to(torch.float64).sum().to(torch.float32)

    return _get_ext().vectorsum_cub_forward(values, output)
