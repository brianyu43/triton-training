from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline

_EXT: Any = None
_STAGE_TIMING_COUNT = 0
_USE_OLD_GATE = os.environ.get("TRIMUL_V40_OLD_GATE", os.environ.get("TRIMUL_V37_OLD_GATE", "")) == "1"
_USE_OLD_HIDDEN = os.environ.get("TRIMUL_V40_OLD_HIDDEN", "") == "1"
_RANK01_WEIGHT_CACHE: Dict[Any, torch.Tensor] = {}
_RANK01_BUFFER_CACHE: Dict[Any, Dict[str, torch.Tensor]] = {}
_C384_WEIGHT_CACHE: Dict[Any, torch.Tensor] = {}
_V48_LR5_CACHE: Dict[Any, torch.Tensor] = {}


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

torch::Tensor trimul_v40_forward(
    torch::Tensor x,
    torch::Tensor mask,
    torch::Tensor norm_weight,
    torch::Tensor norm_bias,
    torch::Tensor left_proj_weight,
    torch::Tensor right_proj_weight,
    torch::Tensor left_gate_weight,
    torch::Tensor right_gate_weight,
    torch::Tensor out_gate_weight,
    torch::Tensor to_out_norm_weight,
    torch::Tensor to_out_norm_bias,
    torch::Tensor to_out_weight,
    int64_t enable_timing,
    int64_t use_old_gate,
    int64_t use_old_hidden
);

torch::Tensor trimul_v48_layernorm(
    torch::Tensor x,
    torch::Tensor norm_weight,
    torch::Tensor norm_bias,
    int64_t hidden_dim
);

torch::Tensor trimul_v48_tail(
    torch::Tensor lr5,
    torch::Tensor to_out_norm_weight,
    torch::Tensor to_out_norm_bias,
    torch::Tensor to_out_weight,
    int64_t enable_timing,
    int64_t use_old_hidden
);

torch::Tensor trimul_v50_pack_rank01_weights(
    torch::Tensor left_proj_weight,
    torch::Tensor right_proj_weight,
    torch::Tensor left_gate_weight,
    torch::Tensor right_gate_weight,
    torch::Tensor out_gate_weight,
    torch::Tensor to_out_weight
);

torch::Tensor trimul_v51_pack5_transpose_weights(
    torch::Tensor left_proj_weight,
    torch::Tensor right_proj_weight,
    torch::Tensor left_gate_weight,
    torch::Tensor right_gate_weight,
    torch::Tensor out_gate_weight
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("trimul_v40_forward", &trimul_v40_forward, "trimul v40 cuda rank02 hidden");
  m.def("trimul_v48_layernorm", &trimul_v48_layernorm, "trimul v48 layernorm only");
  m.def("trimul_v48_tail", &trimul_v48_tail, "trimul v48 tail from H-major lr5");
  m.def("trimul_v50_pack_rank01_weights", &trimul_v50_pack_rank01_weights, "trimul v50 rank01 C128 weight pack");
  m.def("trimul_v51_pack5_transpose_weights", &trimul_v51_pack5_transpose_weights, "trimul v51 C384 projection/gate weight pack");
}
"""

    cuda_src = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <unordered_map>

namespace {

constexpr float kEps = 1.0e-5f;

static void cublas_check(cublasStatus_t status, const char* msg) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, msg);
}

static void cuda_check(cudaError_t status, const char* msg) {
  TORCH_CHECK(status == cudaSuccess, msg, ": ", cudaGetErrorString(status));
}

__device__ __forceinline__ float sigmoid_f(float x) {
  return __fdividef(1.0f, 1.0f + __expf(-x));
}

__device__ __forceinline__ float block_sum(float value) {
  extern __shared__ float shared[];
  const int tid = threadIdx.x;
  shared[tid] = value;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  const float result = shared[0];
  __syncthreads();
  return result;
}

__device__ __forceinline__ float warp_sum(float value) {
  value += __shfl_down_sync(0xffffffff, value, 16);
  value += __shfl_down_sync(0xffffffff, value, 8);
  value += __shfl_down_sync(0xffffffff, value, 4);
  value += __shfl_down_sync(0xffffffff, value, 2);
  value += __shfl_down_sync(0xffffffff, value, 1);
  return value;
}

template <int Dim, int WarpsPerBlock>
__global__ void layernorm_warp_rows_to_f16_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    __half* __restrict__ out,
    int64_t rows) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * WarpsPerBlock + warp;
  if (row >= rows) return;

  const float* x_row = x + row * static_cast<int64_t>(Dim);
  float sum = 0.0f;
  float sumsq = 0.0f;
  for (int c = lane; c < Dim; c += 32) {
    const float v = x_row[c];
    sum += v;
    sumsq = fmaf(v, v, sumsq);
  }

  const float total = __shfl_sync(0xffffffff, warp_sum(sum), 0);
  const float total_sq = __shfl_sync(0xffffffff, warp_sum(sumsq), 0);
  const float mean = total / static_cast<float>(Dim);
  float var = total_sq / static_cast<float>(Dim) - mean * mean;
  var = var < 0.0f ? 0.0f : var;
  const float inv_std = rsqrtf(var + kEps);

  __half* out_row = out + row * static_cast<int64_t>(Dim);
  for (int c = lane; c < Dim; c += 32) {
    const float v = x_row[c];
    const float y = fmaf((v - mean) * inv_std, weight[c], bias[c]);
    out_row[c] = __float2half_rn(y);
  }
}

__global__ void layernorm_rows_to_f16_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    __half* __restrict__ out,
    int64_t rows,
    int dim) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) return;
  const int tid = threadIdx.x;
  const float* x_row = x + row * static_cast<int64_t>(dim);
  float sum = 0.0f;
  float sumsq = 0.0f;
  for (int c = tid; c < dim; c += blockDim.x) {
    const float v = x_row[c];
    sum += v;
    sumsq = fmaf(v, v, sumsq);
  }
  const float total = block_sum(sum);
  const float total_sq = block_sum(sumsq);
  const float mean = total / static_cast<float>(dim);
  float var = total_sq / static_cast<float>(dim) - mean * mean;
  var = var < 0.0f ? 0.0f : var;
  const float inv_std = rsqrtf(var + kEps);
  __half* out_row = out + row * static_cast<int64_t>(dim);
  for (int c = tid; c < dim; c += blockDim.x) {
    const float v = x_row[c];
    const float y = fmaf((v - mean) * inv_std, weight[c], bias[c]);
    out_row[c] = __float2half_rn(y);
  }
}

__global__ void pack6_f32_to_f16_kernel(
    const float* __restrict__ w0,
    const float* __restrict__ w1,
    const float* __restrict__ w2,
    const float* __restrict__ w3,
    const float* __restrict__ w4,
    const float* __restrict__ w5,
    __half* __restrict__ out,
    int64_t seg_elems) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
  const int64_t total = seg_elems * 6;
  if (idx >= total) return;
  const int64_t seg = idx / seg_elems;
  const int64_t off = idx - seg * seg_elems;
  float value = 0.0f;
  if (seg == 0) {
    value = w0[off];
  } else if (seg == 1) {
    value = w1[off];
  } else if (seg == 2) {
    value = w2[off];
  } else if (seg == 3) {
    value = w3[off];
  } else if (seg == 4) {
    value = w4[off];
  } else {
    value = w5[off];
  }
  out[idx] = __float2half_rn(value);
}

__global__ void pack6_rank01_transpose_f32_to_f16_kernel(
    const float* __restrict__ w0,
    const float* __restrict__ w1,
    const float* __restrict__ w2,
    const float* __restrict__ w3,
    const float* __restrict__ w4,
    const float* __restrict__ w5,
    __half* __restrict__ out,
    int64_t dim,
    int64_t hidden_dim) {
  const int64_t seg_elems = dim * hidden_dim;
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
  const int64_t total = seg_elems * 6;
  if (idx >= total) return;

  const int64_t seg = idx / seg_elems;
  const int64_t off = idx - seg * seg_elems;
  float value = 0.0f;
  if (seg < 5) {
    const int64_t c = off / hidden_dim;
    const int64_t h = off - c * hidden_dim;
    const int64_t src = h * dim + c;
    if (seg == 0) {
      value = w0[src];
    } else if (seg == 1) {
      value = w1[src];
    } else if (seg == 2) {
      value = w2[src];
    } else if (seg == 3) {
      value = w3[src];
    } else {
      value = w4[src];
    }
  } else {
    const int64_t h = off / dim;
    const int64_t c = off - h * dim;
    value = w5[c * hidden_dim + h];
  }
  out[idx] = __float2half_rn(value);
}

__global__ void pack5_transpose_f32_to_f16_kernel(
    const float* __restrict__ w0,
    const float* __restrict__ w1,
    const float* __restrict__ w2,
    const float* __restrict__ w3,
    const float* __restrict__ w4,
    __half* __restrict__ out,
    int64_t dim,
    int64_t hidden_dim) {
  const int64_t seg_elems = dim * hidden_dim;
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
  const int64_t total = seg_elems * 5;
  if (idx >= total) return;

  const int64_t seg = idx / seg_elems;
  const int64_t off = idx - seg * seg_elems;
  const int64_t c = off / hidden_dim;
  const int64_t h = off - c * hidden_dim;
  const int64_t src = h * dim + c;
  float value = 0.0f;
  if (seg == 0) {
    value = w0[src];
  } else if (seg == 1) {
    value = w1[src];
  } else if (seg == 2) {
    value = w2[src];
  } else if (seg == 3) {
    value = w3[src];
  } else {
    value = w4[src];
  }
  out[idx] = __float2half_rn(value);
}

__global__ void pack1_f32_to_f16_kernel(
    const float* __restrict__ src,
    __half* __restrict__ out,
    int64_t elems) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
  if (idx >= elems) return;
  out[idx] = __float2half_rn(src[idx]);
}

template <typename MaskT>
__device__ __forceinline__ float mask_value(const MaskT* __restrict__ mask, int64_t row);

template <>
__device__ __forceinline__ float mask_value<float>(const float* __restrict__ mask, int64_t row) {
  return mask[row] == 0.0f ? 0.0f : 1.0f;
}

template <>
__device__ __forceinline__ float mask_value<int64_t>(const int64_t* __restrict__ mask, int64_t row) {
  return mask[row] == 0 ? 0.0f : 1.0f;
}

template <typename MaskT>
__global__ void apply_gate_mask_kernel(
    __half* __restrict__ lr5,
    const MaskT* __restrict__ mask,
    int64_t rows,
    int hidden_dim) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
  const int64_t total = rows * static_cast<int64_t>(hidden_dim);
  if (idx >= total) return;
  const int h = static_cast<int>(idx / rows);
  const int64_t row = idx - static_cast<int64_t>(h) * rows;
  const int64_t plane = rows * static_cast<int64_t>(hidden_dim);
  const float m = mask_value<MaskT>(mask, row);
  const int64_t left_idx = static_cast<int64_t>(h) * rows + row;
  const int64_t right_idx = plane + left_idx;
  const int64_t left_gate_idx = 2 * plane + left_idx;
  const int64_t right_gate_idx = 3 * plane + left_idx;
  const float left = __half2float(lr5[left_idx]);
  const float right = __half2float(lr5[right_idx]);
  const float left_gate = __half2float(lr5[left_gate_idx]);
  const float right_gate = __half2float(lr5[right_gate_idx]);
  lr5[left_idx] = __float2half_rn(left * sigmoid_f(left_gate) * m);
  lr5[right_idx] = __float2half_rn(right * sigmoid_f(right_gate) * m);
}

__device__ __forceinline__ void apply_gate_pair_nomask(
    __half* __restrict__ left,
    __half* __restrict__ right,
    const __half* __restrict__ left_gate,
    const __half* __restrict__ right_gate,
    int64_t off) {
  const __half2 left2 = *reinterpret_cast<const __half2*>(left + off);
  const __half2 right2 = *reinterpret_cast<const __half2*>(right + off);
  const __half2 left_gate2 = *reinterpret_cast<const __half2*>(left_gate + off);
  const __half2 right_gate2 = *reinterpret_cast<const __half2*>(right_gate + off);
  const float2 left_f = __half22float2(left2);
  const float2 right_f = __half22float2(right2);
  const float2 left_gate_f = __half22float2(left_gate2);
  const float2 right_gate_f = __half22float2(right_gate2);
  *reinterpret_cast<__half2*>(left + off) = __floats2half2_rn(
      left_f.x * sigmoid_f(left_gate_f.x),
      left_f.y * sigmoid_f(left_gate_f.y));
  *reinterpret_cast<__half2*>(right + off) = __floats2half2_rn(
      right_f.x * sigmoid_f(right_gate_f.x),
      right_f.y * sigmoid_f(right_gate_f.y));
}

__device__ __forceinline__ void apply_gate_pair_masked(
    __half* __restrict__ left,
    __half* __restrict__ right,
    const __half* __restrict__ left_gate,
    const __half* __restrict__ right_gate,
    int64_t off,
    float mask0,
    float mask1) {
  const __half2 left2 = *reinterpret_cast<const __half2*>(left + off);
  const __half2 right2 = *reinterpret_cast<const __half2*>(right + off);
  const __half2 left_gate2 = *reinterpret_cast<const __half2*>(left_gate + off);
  const __half2 right_gate2 = *reinterpret_cast<const __half2*>(right_gate + off);
  const float2 left_f = __half22float2(left2);
  const float2 right_f = __half22float2(right2);
  const float2 left_gate_f = __half22float2(left_gate2);
  const float2 right_gate_f = __half22float2(right_gate2);
  *reinterpret_cast<__half2*>(left + off) = __floats2half2_rn(
      left_f.x * sigmoid_f(left_gate_f.x) * mask0,
      left_f.y * sigmoid_f(left_gate_f.y) * mask1);
  *reinterpret_cast<__half2*>(right + off) = __floats2half2_rn(
      right_f.x * sigmoid_f(right_gate_f.x) * mask0,
      right_f.y * sigmoid_f(right_gate_f.y) * mask1);
}

__device__ __forceinline__ void apply_gate_scalar_nomask(
    __half* __restrict__ left,
    __half* __restrict__ right,
    const __half* __restrict__ left_gate,
    const __half* __restrict__ right_gate,
    int64_t off) {
  const float left_v = __half2float(left[off]);
  const float right_v = __half2float(right[off]);
  const float left_gate_v = __half2float(left_gate[off]);
  const float right_gate_v = __half2float(right_gate[off]);
  left[off] = __float2half_rn(left_v * sigmoid_f(left_gate_v));
  right[off] = __float2half_rn(right_v * sigmoid_f(right_gate_v));
}

__device__ __forceinline__ void apply_gate_scalar_masked(
    __half* __restrict__ left,
    __half* __restrict__ right,
    const __half* __restrict__ left_gate,
    const __half* __restrict__ right_gate,
    int64_t off,
    float mask_v) {
  const float left_v = __half2float(left[off]);
  const float right_v = __half2float(right[off]);
  const float left_gate_v = __half2float(left_gate[off]);
  const float right_gate_v = __half2float(right_gate[off]);
  left[off] = __float2half_rn(left_v * sigmoid_f(left_gate_v) * mask_v);
  right[off] = __float2half_rn(right_v * sigmoid_f(right_gate_v) * mask_v);
}

template <int BlockThreads>
__global__ __launch_bounds__(BlockThreads, 2) void apply_gate_nomask_vec4_kernel(
    __half* __restrict__ lr5,
    int rows,
    int hidden_dim) {
  const int row = (static_cast<int>(blockIdx.x) * BlockThreads + static_cast<int>(threadIdx.x)) << 2;
  if (row >= rows) return;
  const int h = static_cast<int>(blockIdx.y);
  const int64_t plane = static_cast<int64_t>(hidden_dim) * static_cast<int64_t>(rows);
  const int64_t off = static_cast<int64_t>(h) * static_cast<int64_t>(rows) + static_cast<int64_t>(row);
  __half* left = lr5;
  __half* right = lr5 + plane;
  const __half* left_gate = lr5 + 2 * plane;
  const __half* right_gate = lr5 + 3 * plane;

  if (row + 1 < rows) {
    apply_gate_pair_nomask(left, right, left_gate, right_gate, off);
  } else {
    apply_gate_scalar_nomask(left, right, left_gate, right_gate, off);
    return;
  }
  if (row + 3 < rows) {
    apply_gate_pair_nomask(left, right, left_gate, right_gate, off + 2);
  } else {
    for (int r = row + 2; r < rows; ++r) {
      apply_gate_scalar_nomask(left, right, left_gate, right_gate, off + (r - row));
    }
  }
}

template <int BlockThreads, typename MaskT>
__global__ __launch_bounds__(BlockThreads, 2) void apply_gate_mask_vec4_kernel(
    __half* __restrict__ lr5,
    const MaskT* __restrict__ mask,
    int rows,
    int hidden_dim) {
  const int row = (static_cast<int>(blockIdx.x) * BlockThreads + static_cast<int>(threadIdx.x)) << 2;
  if (row >= rows) return;
  const int h = static_cast<int>(blockIdx.y);
  const int64_t plane = static_cast<int64_t>(hidden_dim) * static_cast<int64_t>(rows);
  const int64_t off = static_cast<int64_t>(h) * static_cast<int64_t>(rows) + static_cast<int64_t>(row);
  __half* left = lr5;
  __half* right = lr5 + plane;
  const __half* left_gate = lr5 + 2 * plane;
  const __half* right_gate = lr5 + 3 * plane;

  const float m0 = mask_value<MaskT>(mask, static_cast<int64_t>(row));
  if (row + 1 < rows) {
    const float m1 = mask_value<MaskT>(mask, static_cast<int64_t>(row + 1));
    apply_gate_pair_masked(left, right, left_gate, right_gate, off, m0, m1);
  } else {
    apply_gate_scalar_masked(left, right, left_gate, right_gate, off, m0);
    return;
  }
  if (row + 3 < rows) {
    const float m2 = mask_value<MaskT>(mask, static_cast<int64_t>(row + 2));
    const float m3 = mask_value<MaskT>(mask, static_cast<int64_t>(row + 3));
    apply_gate_pair_masked(left, right, left_gate, right_gate, off + 2, m2, m3);
  } else {
    for (int r = row + 2; r < rows; ++r) {
      const float mv = mask_value<MaskT>(mask, static_cast<int64_t>(r));
      apply_gate_scalar_masked(left, right, left_gate, right_gate, off + (r - row), mv);
    }
  }
}

__global__ void hidden_ln_gate_to_rows_kernel(
    const __half* __restrict__ hidden_col,
    const __half* __restrict__ out_gate_col,
    const float* __restrict__ norm_weight,
    const float* __restrict__ norm_bias,
    __half* __restrict__ out_rows,
    int64_t rows,
    int hidden_dim) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) return;
  const int tid = threadIdx.x;
  float sum = 0.0f;
  float sumsq = 0.0f;
  for (int h = tid; h < hidden_dim; h += blockDim.x) {
    const float v = __half2float(hidden_col[static_cast<int64_t>(h) * rows + row]);
    sum += v;
    sumsq = fmaf(v, v, sumsq);
  }
  const float total = block_sum(sum);
  const float total_sq = block_sum(sumsq);
  const float mean = total / static_cast<float>(hidden_dim);
  float var = total_sq / static_cast<float>(hidden_dim) - mean * mean;
  var = var < 0.0f ? 0.0f : var;
  const float inv_std = rsqrtf(var + kEps);
  for (int h = tid; h < hidden_dim; h += blockDim.x) {
    const float v = __half2float(hidden_col[static_cast<int64_t>(h) * rows + row]);
    const float gate = sigmoid_f(__half2float(out_gate_col[static_cast<int64_t>(h) * rows + row]));
    const float normed = fmaf((v - mean) * inv_std, norm_weight[h], norm_bias[h]);
    out_rows[row * static_cast<int64_t>(hidden_dim) + h] = __float2half_rn(normed * gate);
  }
}

template <int HiddenDim, int WarpsPerBlock>
__global__ void hidden_ln_gate_to_rows_warp_kernel(
    const __half* __restrict__ hidden_col,
    const __half* __restrict__ out_gate_col,
    const float* __restrict__ norm_weight,
    const float* __restrict__ norm_bias,
    __half* __restrict__ out_rows,
    int64_t rows) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * WarpsPerBlock + warp;
  if (row >= rows) return;

  float sum = 0.0f;
  float sumsq = 0.0f;
  for (int h = lane; h < HiddenDim; h += 32) {
    const float v = __half2float(hidden_col[static_cast<int64_t>(h) * rows + row]);
    sum += v;
    sumsq = fmaf(v, v, sumsq);
  }

  const float total = __shfl_sync(0xffffffff, warp_sum(sum), 0);
  const float total_sq = __shfl_sync(0xffffffff, warp_sum(sumsq), 0);
  const float mean = total / static_cast<float>(HiddenDim);
  float var = total_sq / static_cast<float>(HiddenDim) - mean * mean;
  var = var < 0.0f ? 0.0f : var;
  const float inv_std = rsqrtf(var + kEps);

  __half* out_row = out_rows + row * static_cast<int64_t>(HiddenDim);
  for (int h = lane; h < HiddenDim; h += 32) {
    const float v = __half2float(hidden_col[static_cast<int64_t>(h) * rows + row]);
    const float gate = sigmoid_f(__half2float(out_gate_col[static_cast<int64_t>(h) * rows + row]));
    const float normed = fmaf((v - mean) * inv_std, norm_weight[h], norm_bias[h]);
    out_row[h] = __float2half_rn(normed * gate);
  }
}

template <int HiddenDim, int TileRows>
__global__ void hidden_ln_gate_to_rows_tiled_kernel(
    const __half* __restrict__ hidden_col,
    const __half* __restrict__ out_gate_col,
    const float* __restrict__ norm_weight,
    const float* __restrict__ norm_bias,
    __half* __restrict__ out_rows,
    int64_t rows) {
  extern __shared__ __half smem[];
  __half* hidden_s = smem;
  __half* gate_s = hidden_s + TileRows * HiddenDim;

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t row_start = static_cast<int64_t>(blockIdx.x) * TileRows;
  constexpr int tile_elems = TileRows * HiddenDim;

  for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
    const int h = idx / TileRows;
    const int r = idx - h * TileRows;
    const int64_t row = row_start + r;
    const int dst = r * HiddenDim + h;
    if (row < rows) {
      const int64_t src = static_cast<int64_t>(h) * rows + row;
      hidden_s[dst] = hidden_col[src];
      gate_s[dst] = out_gate_col[src];
    } else {
      hidden_s[dst] = __float2half_rn(0.0f);
      gate_s[dst] = __float2half_rn(0.0f);
    }
  }
  __syncthreads();

  if (warp >= TileRows) return;
  const int64_t row = row_start + warp;
  if (row >= rows) return;

  const int base = warp * HiddenDim;
  float sum = 0.0f;
  float sumsq = 0.0f;
  for (int h = lane; h < HiddenDim; h += 32) {
    const float v = __half2float(hidden_s[base + h]);
    sum += v;
    sumsq = fmaf(v, v, sumsq);
  }

  const float total = __shfl_sync(0xffffffff, warp_sum(sum), 0);
  const float total_sq = __shfl_sync(0xffffffff, warp_sum(sumsq), 0);
  const float mean = total / static_cast<float>(HiddenDim);
  float var = total_sq / static_cast<float>(HiddenDim) - mean * mean;
  var = var < 0.0f ? 0.0f : var;
  const float inv_std = rsqrtf(var + kEps);

  __half* out_row = out_rows + row * static_cast<int64_t>(HiddenDim);
  for (int h = lane; h < HiddenDim; h += 32) {
    const float v = __half2float(hidden_s[base + h]);
    const float gate = sigmoid_f(__half2float(gate_s[base + h]));
    const float normed = fmaf((v - mean) * inv_std, norm_weight[h], norm_bias[h]);
    out_row[h] = __float2half_rn(normed * gate);
  }
}

constexpr int kRank02HiddenTile = 32;
constexpr int kRank02HiddenBlockRows = 8;

template <int HiddenDim>
__global__ void hidden_ln_gate_to_rows_rank02_kernel(
    const __half* __restrict__ hidden_col,
    const __half* __restrict__ out_gate_col,
    const float* __restrict__ norm_weight,
    const float* __restrict__ norm_bias,
    __half* __restrict__ out_rows,
    int64_t rows) {
  __shared__ __half hidden_s[HiddenDim][kRank02HiddenTile + 1];
  __shared__ __half gate_s[HiddenDim][kRank02HiddenTile + 1];

  const int rx = static_cast<int>(threadIdx.x);
  const int ty = static_cast<int>(threadIdx.y);
  const int64_t row_base = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(kRank02HiddenTile);
  const int64_t row = row_base + static_cast<int64_t>(rx);

#pragma unroll
  for (int h = ty; h < HiddenDim; h += kRank02HiddenBlockRows) {
    __half hidden_v = __float2half_rn(0.0f);
    __half gate_v = __float2half_rn(0.0f);
    if (row < rows) {
      const int64_t src = static_cast<int64_t>(h) * rows + row;
      hidden_v = hidden_col[src];
      gate_v = out_gate_col[src];
    }
    hidden_s[h][rx] = hidden_v;
    gate_s[h][rx] = gate_v;
  }

  __syncthreads();

  const int lane = rx;
  const int warp = ty;
  constexpr int kLaneCols = HiddenDim / 32;
  int h_lut[kLaneCols];
  float w_lut[kLaneCols];
  float b_lut[kLaneCols];

#pragma unroll
  for (int t = 0; t < kLaneCols; ++t) {
    const int h = lane + (t << 5);
    h_lut[t] = h;
    w_lut[t] = norm_weight[h];
    b_lut[t] = norm_bias[h];
  }

#pragma unroll
  for (int j = 0; j < kRank02HiddenTile; j += kRank02HiddenBlockRows) {
    const int row_in = warp + j;
    const int64_t row_g = row_base + static_cast<int64_t>(row_in);
    if (row_g >= rows) continue;

    float sum = 0.0f;
    float sumsq = 0.0f;
#pragma unroll
    for (int t = 0; t < kLaneCols; ++t) {
      const int h = h_lut[t];
      const float v = __half2float(hidden_s[h][row_in]);
      sum += v;
      sumsq = fmaf(v, v, sumsq);
    }
    sum = warp_sum(sum);
    sumsq = warp_sum(sumsq);
    float mean = __shfl_sync(0xffffffff, sum, 0) * (1.0f / static_cast<float>(HiddenDim));
    float var = __shfl_sync(0xffffffff, sumsq, 0) * (1.0f / static_cast<float>(HiddenDim)) - mean * mean;
    var = var < 0.0f ? 0.0f : var;
    const float inv_std = rsqrtf(var + kEps);
    mean = __shfl_sync(0xffffffff, mean, 0);
    const float invs = __shfl_sync(0xffffffff, inv_std, 0);

#pragma unroll
    for (int t = 0; t < kLaneCols; ++t) {
      const int h = h_lut[t];
      const float v = __half2float(hidden_s[h][row_in]);
      const float gate = sigmoid_f(__half2float(gate_s[h][row_in]));
      const float normed = fmaf((v - mean) * invs, w_lut[t], b_lut[t]);
      out_rows[row_g * static_cast<int64_t>(HiddenDim) + static_cast<int64_t>(h)] =
          __float2half_rn(normed * gate);
    }
  }
}

static void gemm_f16_abt(
    cublasHandle_t handle,
    const __half* a_rm,
    const __half* b_rm,
    __half* c_rm,
    int64_t m,
    int64_t k,
    int64_t n) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublas_check(
      cublasGemmEx(
          handle,
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          static_cast<int>(n),
          static_cast<int>(m),
          static_cast<int>(k),
          &alpha,
          b_rm,
          CUDA_R_16F,
          static_cast<int>(k),
          a_rm,
          CUDA_R_16F,
          static_cast<int>(k),
          &beta,
          c_rm,
          CUDA_R_16F,
          static_cast<int>(n),
          CUBLAS_COMPUTE_32F_FAST_16F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      "cublasGemmEx failed");
}

static void gemm_strided_batched_f16_abt(
    cublasHandle_t handle,
    const __half* a_rm,
    const __half* b_rm,
    __half* c_rm,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t batch_count,
    int64_t stride_a,
    int64_t stride_b,
    int64_t stride_c) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublas_check(
      cublasGemmStridedBatchedEx(
          handle,
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          static_cast<int>(n),
          static_cast<int>(m),
          static_cast<int>(k),
          &alpha,
          b_rm,
          CUDA_R_16F,
          static_cast<int>(k),
          static_cast<long long>(stride_b),
          a_rm,
          CUDA_R_16F,
          static_cast<int>(k),
          static_cast<long long>(stride_a),
          &beta,
          c_rm,
          CUDA_R_16F,
          static_cast<int>(n),
          static_cast<long long>(stride_c),
          static_cast<int>(batch_count),
          CUBLAS_COMPUTE_32F_FAST_16F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      "cublasGemmStridedBatchedEx failed");
}

struct WorkspaceCache {
  at::Tensor xhat;
  at::Tensor lr5;
  at::Tensor central;
  at::Tensor hidden_rows;
  int64_t bs = -1;
  int64_t n = -1;
  int64_t dim = -1;
  int64_t hidden_dim = -1;
  int64_t rows = -1;
};

struct PackedWeightsCache {
  at::Tensor packed;
  const void* p0 = nullptr;
  const void* p1 = nullptr;
  const void* p2 = nullptr;
  const void* p3 = nullptr;
  const void* p4 = nullptr;
  const void* p5 = nullptr;
  int64_t v0 = -1;
  int64_t v1 = -1;
  int64_t v2 = -1;
  int64_t v3 = -1;
  int64_t v4 = -1;
  int64_t v5 = -1;
  int64_t seg_elems = -1;
};

struct PackedSingleWeightCache {
  at::Tensor packed;
  const void* p = nullptr;
  int64_t v = -1;
  int64_t elems = -1;
};

struct DeviceCaches {
  WorkspaceCache workspace;
  PackedWeightsCache packed_weights;
  PackedSingleWeightCache to_out_weight;
};

static DeviceCaches& get_device_caches(int device) {
  thread_local std::unordered_map<int, DeviceCaches> per_device;
  return per_device[device];
}

static inline int64_t tensor_version(const at::Tensor& t) {
  return static_cast<int64_t>(t._version());
}

}  // namespace

torch::Tensor trimul_v40_forward(
    torch::Tensor x,
    torch::Tensor mask,
    torch::Tensor norm_weight,
    torch::Tensor norm_bias,
    torch::Tensor left_proj_weight,
    torch::Tensor right_proj_weight,
    torch::Tensor left_gate_weight,
    torch::Tensor right_gate_weight,
    torch::Tensor out_gate_weight,
    torch::Tensor to_out_norm_weight,
    torch::Tensor to_out_norm_bias,
    torch::Tensor to_out_weight,
    int64_t enable_timing,
    int64_t use_old_gate,
    int64_t use_old_hidden) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(mask.is_cuda(), "mask must be CUDA");
  TORCH_CHECK(mask.scalar_type() == torch::kFloat32 || mask.scalar_type() == torch::kInt64, "mask dtype");

  const int64_t bs = x.size(0);
  const int64_t n = x.size(1);
  const int64_t dim = x.size(3);
  const int64_t hidden_dim = left_proj_weight.size(0);
  const int64_t rows = bs * n * n;
  TORCH_CHECK(mask.numel() == rows, "mask shape mismatch");

  auto opts_f16 = x.options().dtype(torch::kFloat16);
  DeviceCaches& caches = get_device_caches(x.get_device());
  WorkspaceCache& workspace = caches.workspace;
  const bool workspace_hit =
      workspace.xhat.defined() &&
      workspace.bs == bs &&
      workspace.n == n &&
      workspace.dim == dim &&
      workspace.hidden_dim == hidden_dim &&
      workspace.rows == rows;
  if (!workspace_hit) {
    workspace.xhat = torch::empty({rows, dim}, opts_f16);
    workspace.lr5 = torch::empty({5 * hidden_dim, bs, n, n}, opts_f16);
    workspace.central = torch::empty({hidden_dim, bs, n, n}, opts_f16);
    workspace.hidden_rows = torch::empty({rows, hidden_dim}, opts_f16);
    workspace.bs = bs;
    workspace.n = n;
    workspace.dim = dim;
    workspace.hidden_dim = hidden_dim;
    workspace.rows = rows;
  }
  auto xhat = workspace.xhat;
  auto lr5 = workspace.lr5;
  auto central = workspace.central;
  auto hidden_rows = workspace.hidden_rows;
  auto y = torch::empty({bs, n, n, dim}, opts_f16);

  (void)enable_timing;

  const int ln_threads = 256;
  const size_t shmem = static_cast<size_t>(ln_threads) * sizeof(float);
  constexpr int kLnWarpsPerBlock = 8;
  const unsigned int warp_ln_blocks = static_cast<unsigned int>((rows + kLnWarpsPerBlock - 1) / kLnWarpsPerBlock);
  if (dim == 128) {
    layernorm_warp_rows_to_f16_kernel<128, kLnWarpsPerBlock><<<warp_ln_blocks, ln_threads>>>(
        x.data_ptr<float>(),
        norm_weight.data_ptr<float>(),
        norm_bias.data_ptr<float>(),
        reinterpret_cast<__half*>(xhat.data_ptr<at::Half>()),
        rows);
  } else if (dim == 384) {
    layernorm_warp_rows_to_f16_kernel<384, kLnWarpsPerBlock><<<warp_ln_blocks, ln_threads>>>(
        x.data_ptr<float>(),
        norm_weight.data_ptr<float>(),
        norm_bias.data_ptr<float>(),
        reinterpret_cast<__half*>(xhat.data_ptr<at::Half>()),
        rows);
  } else if (dim == 768) {
    layernorm_warp_rows_to_f16_kernel<768, kLnWarpsPerBlock><<<warp_ln_blocks, ln_threads>>>(
        x.data_ptr<float>(),
        norm_weight.data_ptr<float>(),
        norm_bias.data_ptr<float>(),
        reinterpret_cast<__half*>(xhat.data_ptr<at::Half>()),
        rows);
  } else {
    layernorm_rows_to_f16_kernel<<<static_cast<unsigned int>(rows), ln_threads, shmem>>>(
        x.data_ptr<float>(),
        norm_weight.data_ptr<float>(),
        norm_bias.data_ptr<float>(),
        reinterpret_cast<__half*>(xhat.data_ptr<at::Half>()),
        rows,
        static_cast<int>(dim));
  }
  const int64_t seg_elems = hidden_dim * dim;
  auto packed_weights = torch::empty({6 * seg_elems}, opts_f16);
  const int pack_threads = 256;
  const int pack_blocks = static_cast<int>((6 * seg_elems + pack_threads - 1) / pack_threads);
  pack6_f32_to_f16_kernel<<<pack_blocks, pack_threads>>>(
      left_proj_weight.data_ptr<float>(),
      right_proj_weight.data_ptr<float>(),
      left_gate_weight.data_ptr<float>(),
      right_gate_weight.data_ptr<float>(),
      out_gate_weight.data_ptr<float>(),
      to_out_weight.data_ptr<float>(),
      reinterpret_cast<__half*>(packed_weights.data_ptr<at::Half>()),
      seg_elems);
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  const __half* packed_ptr = reinterpret_cast<const __half*>(packed_weights.data_ptr<at::Half>());
  gemm_f16_abt(
      handle,
      packed_ptr,
      reinterpret_cast<const __half*>(xhat.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(lr5.data_ptr<at::Half>()),
      5 * hidden_dim,
      dim,
      rows);
  const int gate_threads = 256;
  const int64_t gate_total = rows * hidden_dim;
  const int gate_blocks = static_cast<int>((gate_total + gate_threads - 1) / gate_threads);
  constexpr int kVecGateThreads = 256;
  const dim3 vec_gate_blocks(
      static_cast<unsigned int>((rows + 4 * kVecGateThreads - 1) / (4 * kVecGateThreads)),
      static_cast<unsigned int>(hidden_dim));
  if (use_old_gate == 0 && mask.scalar_type() == torch::kFloat32) {
    apply_gate_nomask_vec4_kernel<kVecGateThreads><<<vec_gate_blocks, kVecGateThreads>>>(
        reinterpret_cast<__half*>(lr5.data_ptr<at::Half>()),
        static_cast<int>(rows),
        static_cast<int>(hidden_dim));
  } else if (use_old_gate == 0) {
    apply_gate_mask_vec4_kernel<kVecGateThreads, int64_t><<<vec_gate_blocks, kVecGateThreads>>>(
        reinterpret_cast<__half*>(lr5.data_ptr<at::Half>()),
        mask.data_ptr<int64_t>(),
        static_cast<int>(rows),
        static_cast<int>(hidden_dim));
  } else if (mask.scalar_type() == torch::kFloat32) {
    apply_gate_mask_kernel<float><<<gate_blocks, gate_threads>>>(
        reinterpret_cast<__half*>(lr5.data_ptr<at::Half>()),
        mask.data_ptr<float>(),
        rows,
        static_cast<int>(hidden_dim));
  } else {
    apply_gate_mask_kernel<int64_t><<<gate_blocks, gate_threads>>>(
        reinterpret_cast<__half*>(lr5.data_ptr<at::Half>()),
        mask.data_ptr<int64_t>(),
        rows,
        static_cast<int>(hidden_dim));
  }
  const int64_t stride_mat = n * n;
  const int64_t batch = hidden_dim * bs;
  const __half* left_ptr = reinterpret_cast<const __half*>(lr5.data_ptr<at::Half>());
  const __half* right_ptr = left_ptr + hidden_dim * rows;
  gemm_strided_batched_f16_abt(
      handle,
      left_ptr,
      right_ptr,
      reinterpret_cast<__half*>(central.data_ptr<at::Half>()),
      n,
      n,
      n,
      batch,
      stride_mat,
      stride_mat,
      stride_mat);
  const __half* out_gate_ptr = left_ptr + 4 * hidden_dim * rows;
  if (hidden_dim == 128 && use_old_hidden == 0) {
    dim3 hidden_block(kRank02HiddenTile, kRank02HiddenBlockRows);
    dim3 hidden_grid(static_cast<unsigned int>((rows + kRank02HiddenTile - 1) / kRank02HiddenTile));
    hidden_ln_gate_to_rows_rank02_kernel<128><<<hidden_grid, hidden_block>>>(
        reinterpret_cast<const __half*>(central.data_ptr<at::Half>()),
        out_gate_ptr,
        to_out_norm_weight.data_ptr<float>(),
        to_out_norm_bias.data_ptr<float>(),
        reinterpret_cast<__half*>(hidden_rows.data_ptr<at::Half>()),
        rows);
  } else if (hidden_dim == 128) {
    if (n >= 768) {
      constexpr int kHiddenTileRows = 16;
      constexpr int kHiddenThreads = 512;
      const unsigned int hidden_blocks =
          static_cast<unsigned int>((rows + kHiddenTileRows - 1) / kHiddenTileRows);
      const size_t hidden_shmem = 2 * kHiddenTileRows * 128 * sizeof(__half);
      hidden_ln_gate_to_rows_tiled_kernel<128, kHiddenTileRows><<<hidden_blocks, kHiddenThreads, hidden_shmem>>>(
          reinterpret_cast<const __half*>(central.data_ptr<at::Half>()),
          out_gate_ptr,
          to_out_norm_weight.data_ptr<float>(),
          to_out_norm_bias.data_ptr<float>(),
          reinterpret_cast<__half*>(hidden_rows.data_ptr<at::Half>()),
          rows);
    } else {
      constexpr int kHiddenTileRows = 8;
      constexpr int kHiddenThreads = 256;
      const unsigned int hidden_blocks =
          static_cast<unsigned int>((rows + kHiddenTileRows - 1) / kHiddenTileRows);
      const size_t hidden_shmem = 2 * kHiddenTileRows * 128 * sizeof(__half);
      hidden_ln_gate_to_rows_tiled_kernel<128, kHiddenTileRows><<<hidden_blocks, kHiddenThreads, hidden_shmem>>>(
          reinterpret_cast<const __half*>(central.data_ptr<at::Half>()),
          out_gate_ptr,
          to_out_norm_weight.data_ptr<float>(),
          to_out_norm_bias.data_ptr<float>(),
          reinterpret_cast<__half*>(hidden_rows.data_ptr<at::Half>()),
          rows);
    }
  } else {
    hidden_ln_gate_to_rows_kernel<<<static_cast<unsigned int>(rows), ln_threads, shmem>>>(
        reinterpret_cast<const __half*>(central.data_ptr<at::Half>()),
        out_gate_ptr,
        to_out_norm_weight.data_ptr<float>(),
        to_out_norm_bias.data_ptr<float>(),
        reinterpret_cast<__half*>(hidden_rows.data_ptr<at::Half>()),
        rows,
        static_cast<int>(hidden_dim));
  }
  const __half* to_out_ptr = packed_ptr + 5 * seg_elems;
  gemm_f16_abt(
      handle,
      reinterpret_cast<const __half*>(hidden_rows.data_ptr<at::Half>()),
      to_out_ptr,
      reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
      rows,
      hidden_dim,
      dim);
  return y;
}

torch::Tensor trimul_v48_layernorm(
    torch::Tensor x,
    torch::Tensor norm_weight,
    torch::Tensor norm_bias,
    int64_t hidden_dim) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
  const int64_t bs = x.size(0);
  const int64_t n = x.size(1);
  const int64_t dim = x.size(3);
  const int64_t rows = bs * n * n;

  auto opts_f16 = x.options().dtype(torch::kFloat16);
  DeviceCaches& caches = get_device_caches(x.get_device());
  WorkspaceCache& workspace = caches.workspace;
  const bool workspace_hit =
      workspace.xhat.defined() &&
      workspace.bs == bs &&
      workspace.n == n &&
      workspace.dim == dim &&
      workspace.hidden_dim == hidden_dim &&
      workspace.rows == rows;
  if (!workspace_hit) {
    workspace.xhat = torch::empty({rows, dim}, opts_f16);
    workspace.lr5 = torch::empty({5 * hidden_dim, bs, n, n}, opts_f16);
    workspace.central = torch::empty({hidden_dim, bs, n, n}, opts_f16);
    workspace.hidden_rows = torch::empty({rows, hidden_dim}, opts_f16);
    workspace.bs = bs;
    workspace.n = n;
    workspace.dim = dim;
    workspace.hidden_dim = hidden_dim;
    workspace.rows = rows;
  }

  auto xhat = workspace.xhat;
  const int ln_threads = 256;
  const size_t shmem = static_cast<size_t>(ln_threads) * sizeof(float);
  constexpr int kLnWarpsPerBlock = 8;
  const unsigned int warp_ln_blocks = static_cast<unsigned int>((rows + kLnWarpsPerBlock - 1) / kLnWarpsPerBlock);
  if (dim == 128) {
    layernorm_warp_rows_to_f16_kernel<128, kLnWarpsPerBlock><<<warp_ln_blocks, ln_threads>>>(
        x.data_ptr<float>(),
        norm_weight.data_ptr<float>(),
        norm_bias.data_ptr<float>(),
        reinterpret_cast<__half*>(xhat.data_ptr<at::Half>()),
        rows);
  } else if (dim == 384) {
    layernorm_warp_rows_to_f16_kernel<384, kLnWarpsPerBlock><<<warp_ln_blocks, ln_threads>>>(
        x.data_ptr<float>(),
        norm_weight.data_ptr<float>(),
        norm_bias.data_ptr<float>(),
        reinterpret_cast<__half*>(xhat.data_ptr<at::Half>()),
        rows);
  } else if (dim == 768) {
    layernorm_warp_rows_to_f16_kernel<768, kLnWarpsPerBlock><<<warp_ln_blocks, ln_threads>>>(
        x.data_ptr<float>(),
        norm_weight.data_ptr<float>(),
        norm_bias.data_ptr<float>(),
        reinterpret_cast<__half*>(xhat.data_ptr<at::Half>()),
        rows);
  } else {
    layernorm_rows_to_f16_kernel<<<static_cast<unsigned int>(rows), ln_threads, shmem>>>(
        x.data_ptr<float>(),
        norm_weight.data_ptr<float>(),
        norm_bias.data_ptr<float>(),
        reinterpret_cast<__half*>(xhat.data_ptr<at::Half>()),
        rows,
        static_cast<int>(dim));
  }
  return xhat;
}

torch::Tensor trimul_v48_tail(
    torch::Tensor lr5,
    torch::Tensor to_out_norm_weight,
    torch::Tensor to_out_norm_bias,
    torch::Tensor to_out_weight,
    int64_t enable_timing,
    int64_t use_old_hidden) {
  TORCH_CHECK(lr5.is_cuda(), "lr5 must be CUDA");
  TORCH_CHECK(lr5.scalar_type() == torch::kFloat16, "lr5 must be float16");
  TORCH_CHECK(lr5.dim() == 4, "lr5 shape");

  const int64_t hidden_dim = lr5.size(0) / 5;
  const int64_t bs = lr5.size(1);
  const int64_t n = lr5.size(2);
  const int64_t dim = to_out_weight.size(0);
  const int64_t rows = bs * n * n;
  TORCH_CHECK(lr5.size(0) == 5 * hidden_dim, "lr5 first dim");
  TORCH_CHECK(lr5.size(3) == n, "lr5 square shape");

  auto opts_f16 = lr5.options().dtype(torch::kFloat16);
  DeviceCaches& caches = get_device_caches(lr5.get_device());
  WorkspaceCache& workspace = caches.workspace;
  const bool workspace_hit =
      workspace.central.defined() &&
      workspace.bs == bs &&
      workspace.n == n &&
      workspace.dim == dim &&
      workspace.hidden_dim == hidden_dim &&
      workspace.rows == rows;
  if (!workspace_hit) {
    workspace.xhat = torch::empty({rows, dim}, opts_f16);
    workspace.lr5 = torch::empty({5 * hidden_dim, bs, n, n}, opts_f16);
    workspace.central = torch::empty({hidden_dim, bs, n, n}, opts_f16);
    workspace.hidden_rows = torch::empty({rows, hidden_dim}, opts_f16);
    workspace.bs = bs;
    workspace.n = n;
    workspace.dim = dim;
    workspace.hidden_dim = hidden_dim;
    workspace.rows = rows;
  }
  auto central = workspace.central;
  auto hidden_rows = workspace.hidden_rows;
  auto y = torch::empty({bs, n, n, dim}, opts_f16);

  (void)enable_timing;

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  const int64_t stride_mat = n * n;
  const int64_t batch = hidden_dim * bs;
  const __half* left_ptr = reinterpret_cast<const __half*>(lr5.data_ptr<at::Half>());
  const __half* right_ptr = left_ptr + hidden_dim * rows;
  gemm_strided_batched_f16_abt(
      handle,
      left_ptr,
      right_ptr,
      reinterpret_cast<__half*>(central.data_ptr<at::Half>()),
      n,
      n,
      n,
      batch,
      stride_mat,
      stride_mat,
      stride_mat);
  const int ln_threads = 256;
  const size_t shmem = static_cast<size_t>(ln_threads) * sizeof(float);
  const __half* out_gate_ptr = left_ptr + 4 * hidden_dim * rows;
  if (hidden_dim == 128 && use_old_hidden == 0) {
    dim3 hidden_block(kRank02HiddenTile, kRank02HiddenBlockRows);
    dim3 hidden_grid(static_cast<unsigned int>((rows + kRank02HiddenTile - 1) / kRank02HiddenTile));
    hidden_ln_gate_to_rows_rank02_kernel<128><<<hidden_grid, hidden_block>>>(
        reinterpret_cast<const __half*>(central.data_ptr<at::Half>()),
        out_gate_ptr,
        to_out_norm_weight.data_ptr<float>(),
        to_out_norm_bias.data_ptr<float>(),
        reinterpret_cast<__half*>(hidden_rows.data_ptr<at::Half>()),
        rows);
  } else {
    hidden_ln_gate_to_rows_kernel<<<static_cast<unsigned int>(rows), ln_threads, shmem>>>(
        reinterpret_cast<const __half*>(central.data_ptr<at::Half>()),
        out_gate_ptr,
        to_out_norm_weight.data_ptr<float>(),
        to_out_norm_bias.data_ptr<float>(),
        reinterpret_cast<__half*>(hidden_rows.data_ptr<at::Half>()),
        rows,
        static_cast<int>(hidden_dim));
  }
  const int64_t out_elems = hidden_dim * dim;
  auto packed_to_out = torch::empty({out_elems}, opts_f16);
  const int pack_threads = 256;
  const int pack_blocks = static_cast<int>((out_elems + pack_threads - 1) / pack_threads);
  pack1_f32_to_f16_kernel<<<pack_blocks, pack_threads>>>(
      to_out_weight.data_ptr<float>(),
      reinterpret_cast<__half*>(packed_to_out.data_ptr<at::Half>()),
      out_elems);

  gemm_f16_abt(
      handle,
      reinterpret_cast<const __half*>(hidden_rows.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(packed_to_out.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
      rows,
      hidden_dim,
      dim);
  return y;
}

torch::Tensor trimul_v50_pack_rank01_weights(
    torch::Tensor left_proj_weight,
    torch::Tensor right_proj_weight,
    torch::Tensor left_gate_weight,
    torch::Tensor right_gate_weight,
    torch::Tensor out_gate_weight,
    torch::Tensor to_out_weight) {
  TORCH_CHECK(left_proj_weight.is_cuda(), "left_proj_weight must be CUDA");
  TORCH_CHECK(right_proj_weight.is_cuda(), "right_proj_weight must be CUDA");
  TORCH_CHECK(left_gate_weight.is_cuda(), "left_gate_weight must be CUDA");
  TORCH_CHECK(right_gate_weight.is_cuda(), "right_gate_weight must be CUDA");
  TORCH_CHECK(out_gate_weight.is_cuda(), "out_gate_weight must be CUDA");
  TORCH_CHECK(to_out_weight.is_cuda(), "to_out_weight must be CUDA");
  TORCH_CHECK(left_proj_weight.scalar_type() == torch::kFloat32, "rank01 weights must be float32");
  TORCH_CHECK(right_proj_weight.scalar_type() == torch::kFloat32, "rank01 weights must be float32");
  TORCH_CHECK(left_gate_weight.scalar_type() == torch::kFloat32, "rank01 weights must be float32");
  TORCH_CHECK(right_gate_weight.scalar_type() == torch::kFloat32, "rank01 weights must be float32");
  TORCH_CHECK(out_gate_weight.scalar_type() == torch::kFloat32, "rank01 weights must be float32");
  TORCH_CHECK(to_out_weight.scalar_type() == torch::kFloat32, "rank01 weights must be float32");
  TORCH_CHECK(left_proj_weight.is_contiguous(), "left_proj_weight must be contiguous");
  TORCH_CHECK(right_proj_weight.is_contiguous(), "right_proj_weight must be contiguous");
  TORCH_CHECK(left_gate_weight.is_contiguous(), "left_gate_weight must be contiguous");
  TORCH_CHECK(right_gate_weight.is_contiguous(), "right_gate_weight must be contiguous");
  TORCH_CHECK(out_gate_weight.is_contiguous(), "out_gate_weight must be contiguous");
  TORCH_CHECK(to_out_weight.is_contiguous(), "to_out_weight must be contiguous");

  const int64_t hidden_dim = left_proj_weight.size(0);
  const int64_t dim = left_proj_weight.size(1);
  TORCH_CHECK(right_proj_weight.size(0) == hidden_dim && right_proj_weight.size(1) == dim, "right_proj shape");
  TORCH_CHECK(left_gate_weight.size(0) == hidden_dim && left_gate_weight.size(1) == dim, "left_gate shape");
  TORCH_CHECK(right_gate_weight.size(0) == hidden_dim && right_gate_weight.size(1) == dim, "right_gate shape");
  TORCH_CHECK(out_gate_weight.size(0) == hidden_dim && out_gate_weight.size(1) == dim, "out_gate shape");
  TORCH_CHECK(to_out_weight.size(0) == dim && to_out_weight.size(1) == hidden_dim, "to_out shape");

  const int64_t seg_elems = dim * hidden_dim;
  auto packed = torch::empty({6 * seg_elems}, left_proj_weight.options().dtype(torch::kFloat16));
  const int threads = 256;
  const int blocks = static_cast<int>((6 * seg_elems + threads - 1) / threads);
  pack6_rank01_transpose_f32_to_f16_kernel<<<blocks, threads>>>(
      left_proj_weight.data_ptr<float>(),
      right_proj_weight.data_ptr<float>(),
      left_gate_weight.data_ptr<float>(),
      right_gate_weight.data_ptr<float>(),
      out_gate_weight.data_ptr<float>(),
      to_out_weight.data_ptr<float>(),
      reinterpret_cast<__half*>(packed.data_ptr<at::Half>()),
      dim,
      hidden_dim);
  return packed;
}

torch::Tensor trimul_v51_pack5_transpose_weights(
    torch::Tensor left_proj_weight,
    torch::Tensor right_proj_weight,
    torch::Tensor left_gate_weight,
    torch::Tensor right_gate_weight,
    torch::Tensor out_gate_weight) {
  TORCH_CHECK(left_proj_weight.is_cuda(), "left_proj_weight must be CUDA");
  TORCH_CHECK(right_proj_weight.is_cuda(), "right_proj_weight must be CUDA");
  TORCH_CHECK(left_gate_weight.is_cuda(), "left_gate_weight must be CUDA");
  TORCH_CHECK(right_gate_weight.is_cuda(), "right_gate_weight must be CUDA");
  TORCH_CHECK(out_gate_weight.is_cuda(), "out_gate_weight must be CUDA");
  TORCH_CHECK(left_proj_weight.scalar_type() == torch::kFloat32, "projection weights must be float32");
  TORCH_CHECK(right_proj_weight.scalar_type() == torch::kFloat32, "projection weights must be float32");
  TORCH_CHECK(left_gate_weight.scalar_type() == torch::kFloat32, "projection weights must be float32");
  TORCH_CHECK(right_gate_weight.scalar_type() == torch::kFloat32, "projection weights must be float32");
  TORCH_CHECK(out_gate_weight.scalar_type() == torch::kFloat32, "projection weights must be float32");
  TORCH_CHECK(left_proj_weight.is_contiguous(), "left_proj_weight must be contiguous");
  TORCH_CHECK(right_proj_weight.is_contiguous(), "right_proj_weight must be contiguous");
  TORCH_CHECK(left_gate_weight.is_contiguous(), "left_gate_weight must be contiguous");
  TORCH_CHECK(right_gate_weight.is_contiguous(), "right_gate_weight must be contiguous");
  TORCH_CHECK(out_gate_weight.is_contiguous(), "out_gate_weight must be contiguous");

  const int64_t hidden_dim = left_proj_weight.size(0);
  const int64_t dim = left_proj_weight.size(1);
  TORCH_CHECK(right_proj_weight.size(0) == hidden_dim && right_proj_weight.size(1) == dim, "right_proj shape");
  TORCH_CHECK(left_gate_weight.size(0) == hidden_dim && left_gate_weight.size(1) == dim, "left_gate shape");
  TORCH_CHECK(right_gate_weight.size(0) == hidden_dim && right_gate_weight.size(1) == dim, "right_gate shape");
  TORCH_CHECK(out_gate_weight.size(0) == hidden_dim && out_gate_weight.size(1) == dim, "out_gate shape");

  const int64_t seg_elems = dim * hidden_dim;
  auto packed = torch::empty({5 * seg_elems}, left_proj_weight.options().dtype(torch::kFloat16));
  const int threads = 256;
  const int blocks = static_cast<int>((5 * seg_elems + threads - 1) / threads);
  pack5_transpose_f32_to_f16_kernel<<<blocks, threads>>>(
      left_proj_weight.data_ptr<float>(),
      right_proj_weight.data_ptr<float>(),
      left_gate_weight.data_ptr<float>(),
      right_gate_weight.data_ptr<float>(),
      out_gate_weight.data_ptr<float>(),
      reinterpret_cast<__half*>(packed.data_ptr<at::Half>()),
      dim,
      hidden_dim);
  return packed;
}
"""

    _EXT = load_inline(
        name="trimul_v51_c384_tail_fused",
        cpp_sources=cpp_src,
        cuda_sources=cuda_src,
        functions=None,
        with_cuda=True,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-arch=sm_80",
            "--use_fast_math",
        ],
        verbose=False,
    )
    return _EXT


@torch.no_grad()
def _v40_path(data: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]) -> torch.Tensor:
    x, mask, weights, _ = data
    ext = _get_ext()
    return ext.trimul_v40_forward(
        x,
        mask,
        weights["norm.weight"],
        weights["norm.bias"],
        weights["left_proj.weight"],
        weights["right_proj.weight"],
        weights["left_gate.weight"],
        weights["right_gate.weight"],
        weights["out_gate.weight"],
        weights["to_out_norm.weight"],
        weights["to_out_norm.bias"],
        weights["to_out.weight"],
        0,
        int(_USE_OLD_GATE),
        int(_USE_OLD_HIDDEN),
    )


# Rank01 Triton path used only for large/mid C128 benchmark shapes.
# ----------------------------------------------------------------------
# 1) Row‑wise LayerNorm (fp16 out, fp32 accumulator)
# ----------------------------------------------------------------------
@triton.jit
def _row_ln_fp16_kernel(
    X_ptr, Y_ptr,          # (M, C) input / output
    w_ptr, b_ptr,          # LN weight & bias (fp32)
    M, C: tl.constexpr,    # rows, columns (C must be constexpr)
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    row_mask = rows < M

    # ---------- compute mean & variance (fp32) ----------
    sum_val = tl.zeros([BLOCK_M], dtype=tl.float32)
    sumsq_val = tl.zeros([BLOCK_M], dtype=tl.float32)

    for c in range(0, C, BLOCK_C):
        cur_c = c + tl.arange(0, BLOCK_C)
        col_mask = cur_c < C
        x = tl.load(
            X_ptr + rows[:, None] * C + cur_c[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)                     # (BLOCK_M, BLOCK_C)
        sum_val += tl.sum(x, axis=1)
        sumsq_val += tl.sum(x * x, axis=1)

    mean = sum_val / C
    var = sumsq_val / C - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # ---------- normalize + affine (fp16) ----------
    for c in range(0, C, BLOCK_C):
        cur_c = c + tl.arange(0, BLOCK_C)
        col_mask = cur_c < C
        x = tl.load(
            X_ptr + rows[:, None] * C + cur_c[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        y = (x - mean[:, None]) * inv_std[:, None]

        w = tl.load(w_ptr + cur_c, mask=col_mask, other=0.0)
        b = tl.load(b_ptr + cur_c, mask=col_mask, other=0.0)

        y = y * w[None, :] + b[None, :]
        tl.store(
            Y_ptr + rows[:, None] * C + cur_c[None, :],
            y.to(tl.float16),
            mask=row_mask[:, None] & col_mask[None, :],
        )


def _row_layernorm_fp16(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Row‑wise LayerNorm over the last dim → FP16 output."""
    B, N, _, C = x.shape
    M = B * N * N
    x_flat = x.view(M, C).contiguous()
    if out is None:
        y_flat = torch.empty((M, C), dtype=torch.float16, device=x.device)
    else:
        y_flat = out.view(M, C)

    BLOCK_M = 128
    BLOCK_C = 128
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)

    _row_ln_fp16_kernel[grid](
        x_flat,
        y_flat,
        weight,
        bias,
        M,
        C,
        eps,
        BLOCK_M=BLOCK_M,
        BLOCK_C=BLOCK_C,
        num_warps=8,
    )
    return y_flat.view(B, N, N, C)


# ----------------------------------------------------------------------
# 2) Fused projection + gating + optional scalar mask
# ----------------------------------------------------------------------
@triton.jit
def _proj_gate_mask_kernel(
    x_ptr,                         # (M, C) fp16
    mask_ptr,                      # (M,) fp16  (if MASKED==1)
    left_proj_w_ptr,               # (C, H) fp16
    left_gate_w_ptr,               # (C, H) fp16
    right_proj_w_ptr,              # (C, H) fp16
    right_gate_w_ptr,              # (C, H) fp16
    out_gate_w_ptr,                # (C, H) fp16
    left_ptr,                      # (B, H, N, N) fp16
    right_ptr,                     # (B, H, N, N) fp16
    out_gate_ptr,                  # (B, N, N, H) fp16
    M, N, C: tl.constexpr, H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MASKED: tl.constexpr,
):
    pid_m = tl.program_id(0)  # row block
    pid_h = tl.program_id(1)  # hidden block

    row_start = pid_m * BLOCK_M
    hid_start = pid_h * BLOCK_H

    rows = row_start + tl.arange(0, BLOCK_M)          # (BLOCK_M,)
    hids = hid_start + tl.arange(0, BLOCK_H)         # (BLOCK_H,)

    row_mask = rows < M
    hid_mask = hids < H

    # ---- scalar mask per row (if any) ----
    if MASKED:
        mask_val = tl.load(mask_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)  # (BLOCK_M,)
    else:
        mask_val = tl.full([BLOCK_M], 1.0, dtype=tl.float32)

    # ---- accumulators (fp32) ----
    acc_lp = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)  # left proj
    acc_lg = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)  # left gate
    acc_rp = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)  # right proj
    acc_rg = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)  # right gate
    acc_og = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)  # out gate

    for k in range(0, C, BLOCK_K):
        cur_k = k + tl.arange(0, BLOCK_K)
        k_mask = cur_k < C

        # input tile (fp16 → fp32)
        a = tl.load(
            x_ptr + rows[:, None] * C + cur_k[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )  # (BLOCK_M, BLOCK_K) fp16

        # weight tiles (C,H) column‑major
        w_lp = tl.load(left_proj_w_ptr + cur_k[:, None] * H + hids[None, :],
                       mask=k_mask[:, None] & hid_mask[None, :],
                       other=0.0)
        w_lg = tl.load(left_gate_w_ptr + cur_k[:, None] * H + hids[None, :],
                       mask=k_mask[:, None] & hid_mask[None, :],
                       other=0.0)
        w_rp = tl.load(right_proj_w_ptr + cur_k[:, None] * H + hids[None, :],
                       mask=k_mask[:, None] & hid_mask[None, :],
                       other=0.0)
        w_rg = tl.load(right_gate_w_ptr + cur_k[:, None] * H + hids[None, :],
                       mask=k_mask[:, None] & hid_mask[None, :],
                       other=0.0)
        w_og = tl.load(out_gate_w_ptr + cur_k[:, None] * H + hids[None, :],
                       mask=k_mask[:, None] & hid_mask[None, :],
                       other=0.0)

        # fp16·fp16 → fp32 dot products
        acc_lp += tl.dot(a, w_lp)
        acc_lg += tl.dot(a, w_lg)
        acc_rp += tl.dot(a, w_rp)
        acc_rg += tl.dot(a, w_rg)
        acc_og += tl.dot(a, w_og)

    # ---- sigmoid (fp32) ----
    left_gate  = 1.0 / (1.0 + tl.exp(-acc_lg))
    right_gate = 1.0 / (1.0 + tl.exp(-acc_rg))
    out_gate   = 1.0 / (1.0 + tl.exp(-acc_og))

    # ---- apply mask and per‑row gates ----
    left_out  = acc_lp * left_gate * mask_val[:, None]
    right_out = acc_rp * right_gate * mask_val[:, None]

    # ---- map flat row index (b,i,k) → coordinates ----
    N_sq = N * N
    b_idx = rows // N_sq
    rem   = rows - b_idx * N_sq
    i_idx = rem // N
    k_idx = rem - i_idx * N

    # layout for left/right: (B, H, N, N)
    left_offset = ((b_idx[:, None] * H + hids[None, :]) * N_sq) + i_idx[:, None] * N + k_idx[:, None]

    tl.store(
        left_ptr + left_offset,
        left_out.to(tl.float16),
        mask=row_mask[:, None] & hid_mask[None, :],
    )
    tl.store(
        right_ptr + left_offset,
        right_out.to(tl.float16),
        mask=row_mask[:, None] & hid_mask[None, :],
    )

    # out_gate layout: (B, N, N, H)
    out_gate_offset = rows[:, None] * H + hids[None, :]
    tl.store(
        out_gate_ptr + out_gate_offset,
        out_gate.to(tl.float16),
        mask=row_mask[:, None] & hid_mask[None, :],
    )


@triton.jit
def _proj_gate_mask_hmajor_kernel(
    x_ptr,                         # (M, C) fp16
    mask_ptr,                      # (M,) fp16  (if MASKED==1)
    left_proj_w_ptr,               # (C, H) fp16
    left_gate_w_ptr,               # (C, H) fp16
    right_proj_w_ptr,              # (C, H) fp16
    right_gate_w_ptr,              # (C, H) fp16
    out_gate_w_ptr,                # (C, H) fp16
    lr5_ptr,                       # (5, H, M) fp16 flattened
    M, C: tl.constexpr, H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MASKED: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    hids = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    row_mask = rows < M
    hid_mask = hids < H

    if MASKED:
        mask_val = tl.load(mask_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    else:
        mask_val = tl.full([BLOCK_M], 1.0, dtype=tl.float32)

    acc_lp = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
    acc_lg = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
    acc_rp = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
    acc_rg = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
    acc_og = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)

    for k in range(0, C, BLOCK_K):
        cur_k = k + tl.arange(0, BLOCK_K)
        k_mask = cur_k < C
        a = tl.load(
            x_ptr + rows[:, None] * C + cur_k[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        w_lp = tl.load(
            left_proj_w_ptr + cur_k[:, None] * H + hids[None, :],
            mask=k_mask[:, None] & hid_mask[None, :],
            other=0.0,
        )
        w_lg = tl.load(
            left_gate_w_ptr + cur_k[:, None] * H + hids[None, :],
            mask=k_mask[:, None] & hid_mask[None, :],
            other=0.0,
        )
        w_rp = tl.load(
            right_proj_w_ptr + cur_k[:, None] * H + hids[None, :],
            mask=k_mask[:, None] & hid_mask[None, :],
            other=0.0,
        )
        w_rg = tl.load(
            right_gate_w_ptr + cur_k[:, None] * H + hids[None, :],
            mask=k_mask[:, None] & hid_mask[None, :],
            other=0.0,
        )
        w_og = tl.load(
            out_gate_w_ptr + cur_k[:, None] * H + hids[None, :],
            mask=k_mask[:, None] & hid_mask[None, :],
            other=0.0,
        )

        acc_lp += tl.dot(a, w_lp)
        acc_lg += tl.dot(a, w_lg)
        acc_rp += tl.dot(a, w_rp)
        acc_rg += tl.dot(a, w_rg)
        acc_og += tl.dot(a, w_og)

    left_gate = 1.0 / (1.0 + tl.exp(-acc_lg))
    right_gate = 1.0 / (1.0 + tl.exp(-acc_rg))
    plane = M * H
    hmajor_off = hids[None, :] * M + rows[:, None]
    mask = row_mask[:, None] & hid_mask[None, :]
    tl.store(lr5_ptr + hmajor_off, (acc_lp * left_gate * mask_val[:, None]).to(tl.float16), mask=mask)
    tl.store(lr5_ptr + plane + hmajor_off, (acc_rp * right_gate * mask_val[:, None]).to(tl.float16), mask=mask)
    tl.store(lr5_ptr + 4 * plane + hmajor_off, acc_og.to(tl.float16), mask=mask)


# ----------------------------------------------------------------------
# 3) Fused hidden‑dim LayerNorm → out‑gate → final linear
# ----------------------------------------------------------------------
@triton.jit
def _ln_gate_out_linear_fused_kernel(
    hidden_ptr,           # (B*H*N*N,) fp16 flattened
    out_gate_ptr,         # (B*N*N*H,) fp16 flattened
    ln_w_ptr, ln_b_ptr,  # (H,) fp32
    w_out_ptr,            # (H, D) fp16
    out_ptr,              # (B, N, N, D) fp32
    B, N, H, D: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)                # flat index for (b,i,j)
    row_mask = rows < (B * N * N)

    N_sq = N * N
    b_idx = rows // N_sq
    rem = rows - b_idx * N_sq
    i_idx = rem // N
    j_idx = rem - i_idx * N

    # hidden tile (BLOCK_M, BLOCK_H)
    hids = tl.arange(0, BLOCK_H)
    hid_mask = hids < H

    hidden_off = ((b_idx[:, None] * H + hids[None, :]) * N_sq) + i_idx[:, None] * N + j_idx[:, None]
    hidden_tile = tl.load(
        hidden_ptr + hidden_off,
        mask=row_mask[:, None] & hid_mask[None, :],
        other=0.0,
    )  # fp16

    hidden_fp32 = hidden_tile.to(tl.float32)

    # ---- mean / variance across H (fp32) ----
    sum_val = tl.sum(hidden_fp32, axis=1)                     # (BLOCK_M,)
    sumsq_val = tl.sum(hidden_fp32 * hidden_fp32, axis=1)    # (BLOCK_M,)
    mean = sum_val / H
    var = sumsq_val / H - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)                       # (BLOCK_M,)

    # ---- layer‑norm (fp32) ----
    w_ln = tl.load(ln_w_ptr + hids, mask=hid_mask, other=0.0)  # (H,)
    b_ln = tl.load(ln_b_ptr + hids, mask=hid_mask, other=0.0)  # (H,)
    hidden_norm = (hidden_fp32 - mean[:, None]) * inv_std[:, None]
    hidden_norm = hidden_norm * w_ln[None, :] + b_ln[None, :]   # (BLOCK_M, BLOCK_H)

    # ---- out‑gate (fp32) ----
    out_gate_off = rows[:, None] * H + hids[None, :]
    out_gate_tile = tl.load(
        out_gate_ptr + out_gate_off,
        mask=row_mask[:, None] & hid_mask[None, :],
        other=0.0,
    ).to(tl.float32)                                          # (BLOCK_M, BLOCK_H)

    gated = hidden_norm * out_gate_tile                        # (BLOCK_M, BLOCK_H)

    # Convert to fp16 for the final matrix‑multiply (TensorCore friendly)
    gated_fp16 = gated.to(tl.float16)

    # ---- final linear projection (fp32) ----
    for d0 in range(0, D, BLOCK_D):
        cols = d0 + tl.arange(0, BLOCK_D)
        col_mask = cols < D
        w_out = tl.load(
            w_out_ptr + hids[:, None] * D + cols[None, :],
            mask=hid_mask[:, None] & col_mask[None, :],
            other=0.0,
        )  # (BLOCK_H, BLOCK_D) fp16

        out = tl.dot(gated_fp16, w_out)                      # (BLOCK_M, BLOCK_D) fp32

        tl.store(
            out_ptr + rows[:, None] * D + cols[None, :],
            out,
            mask=row_mask[:, None] & col_mask[None, :],
        )


# ----------------------------------------------------------------------
# 4) Entry point
# ----------------------------------------------------------------------
def _rank01_weight_token(t: torch.Tensor) -> Tuple[int, int, Tuple[int, ...], Tuple[int, ...]]:
    return (
        t.data_ptr(),
        int(t._version),
        tuple(t.shape),
        tuple(t.stride()),
    )


def _rank01_weights_fp16(
    weights: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    def views_from_packed(packed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dim = int(weights["left_proj.weight"].shape[1])
        hidden_dim = int(weights["left_proj.weight"].shape[0])
        seg = dim * hidden_dim
        return (
            packed[0 * seg : 1 * seg].view(dim, hidden_dim),
            packed[1 * seg : 2 * seg].view(dim, hidden_dim),
            packed[2 * seg : 3 * seg].view(dim, hidden_dim),
            packed[3 * seg : 4 * seg].view(dim, hidden_dim),
            packed[4 * seg : 5 * seg].view(dim, hidden_dim),
            packed[5 * seg : 6 * seg].view(hidden_dim, dim),
        )

    def pack_once() -> torch.Tensor:
        return _get_ext().trimul_v50_pack_rank01_weights(
            weights["left_proj.weight"],
            weights["right_proj.weight"],
            weights["left_gate.weight"],
            weights["right_gate.weight"],
            weights["out_gate.weight"],
            weights["to_out.weight"],
        )

    if os.environ.get("TRIMUL_V50_ENABLE_RANK01_WEIGHT_CACHE", "") != "1":
        return views_from_packed(pack_once())

    names = (
        "left_proj.weight",
        "right_proj.weight",
        "left_gate.weight",
        "right_gate.weight",
        "out_gate.weight",
        "to_out.weight",
    )
    key = tuple(_rank01_weight_token(weights[name]) for name in names)
    cached = _RANK01_WEIGHT_CACHE.get(key)
    if cached is not None:
        return views_from_packed(cached)

    cached = pack_once()
    _RANK01_WEIGHT_CACHE[key] = cached
    return views_from_packed(cached)


def _rank01_buffers(
    device: torch.device,
    B: int,
    N: int,
    C: int,
    H: int,
) -> Dict[str, torch.Tensor]:
    if os.environ.get("TRIMUL_V50_DISABLE_RANK01_BUFFER_CACHE", "") == "1":
        return {
            "x_norm": torch.empty((B, N, N, C), dtype=torch.float16, device=device),
            "left": torch.empty((B, H, N, N), dtype=torch.float16, device=device),
            "right": torch.empty((B, H, N, N), dtype=torch.float16, device=device),
            "out_gate": torch.empty((B, N, N, H), dtype=torch.float16, device=device),
            "hidden": torch.empty((B * H, N, N), dtype=torch.float16, device=device),
            "empty_mask": torch.empty(0, dtype=torch.float16, device=device),
        }

    key = (device.type, device.index, B, N, C, H)
    cached = _RANK01_BUFFER_CACHE.get(key)
    if cached is not None:
        return cached

    cached = {
        "x_norm": torch.empty((B, N, N, C), dtype=torch.float16, device=device),
        "left": torch.empty((B, H, N, N), dtype=torch.float16, device=device),
        "right": torch.empty((B, H, N, N), dtype=torch.float16, device=device),
        "out_gate": torch.empty((B, N, N, H), dtype=torch.float16, device=device),
        "hidden": torch.empty((B * H, N, N), dtype=torch.float16, device=device),
        "empty_mask": torch.empty(0, dtype=torch.float16, device=device),
    }
    _RANK01_BUFFER_CACHE[key] = cached
    return cached


def _packed5_weight_token(t: torch.Tensor) -> Tuple[int, int, Tuple[int, ...], Tuple[int, ...]]:
    return (
        t.data_ptr(),
        int(t._version),
        tuple(t.shape),
        tuple(t.stride()),
    )


def _c384_proj_gate_weights_fp16(
    weights: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    def views_from_packed(packed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dim = int(weights["left_proj.weight"].shape[1])
        hidden_dim = int(weights["left_proj.weight"].shape[0])
        seg = dim * hidden_dim
        return (
            packed[0 * seg : 1 * seg].view(dim, hidden_dim),
            packed[1 * seg : 2 * seg].view(dim, hidden_dim),
            packed[2 * seg : 3 * seg].view(dim, hidden_dim),
            packed[3 * seg : 4 * seg].view(dim, hidden_dim),
            packed[4 * seg : 5 * seg].view(dim, hidden_dim),
        )

    def pack_once() -> torch.Tensor:
        return _get_ext().trimul_v51_pack5_transpose_weights(
            weights["left_proj.weight"],
            weights["right_proj.weight"],
            weights["left_gate.weight"],
            weights["right_gate.weight"],
            weights["out_gate.weight"],
        )

    if os.environ.get("TRIMUL_V51_ENABLE_C384_WEIGHT_CACHE", "") != "1":
        return views_from_packed(pack_once())

    names = (
        "left_proj.weight",
        "right_proj.weight",
        "left_gate.weight",
        "right_gate.weight",
        "out_gate.weight",
    )
    key = tuple(_packed5_weight_token(weights[name]) for name in names)
    cached = _C384_WEIGHT_CACHE.get(key)
    if cached is not None:
        return views_from_packed(cached)

    cached = pack_once()
    _C384_WEIGHT_CACHE[key] = cached
    return views_from_packed(cached)


def _v48_lr5_buffer(
    device: torch.device,
    B: int,
    N: int,
    C: int,
    H: int,
) -> torch.Tensor:
    if os.environ.get("TRIMUL_V51_DISABLE_LR5_CACHE", "") == "1":
        return torch.empty((5 * H, B, N, N), dtype=torch.float16, device=device)

    key = (device.type, device.index, B, N, C, H)
    cached = _V48_LR5_CACHE.get(key)
    if cached is not None:
        return cached

    cached = torch.empty((5 * H, B, N, N), dtype=torch.float16, device=device)
    _V48_LR5_CACHE[key] = cached
    return cached


def _rank01_c128_path(
    data: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict]
) -> torch.Tensor:
    """
    Forward pass of the outgoing TriMul operator (no gradients).

    Parameters
    ----------
    data : tuple
        (input, mask, weights, config)
        - input : Tensor[B, N, N, C]   (float32)
        - mask  : Tensor[B, N, N]      (bool/float) or None
        - weights: dict of module parameters (float32)
        - config : dict with ``dim`` (C) and ``hidden_dim`` (H) and optional ``nomask``

    Returns
    -------
    Tensor[B, N, N, C] (float32)
    """
    # --------------------------------------------------------------
    # unpack arguments
    # --------------------------------------------------------------
    inp, mask, weights, cfg = data
    dim = cfg["dim"]                     # C
    hidden_dim = cfg["hidden_dim"]       # H
    cfg_nomask = cfg.get("nomask", None)
    if cfg_nomask is None:
        nomask = mask is None or mask.dtype == torch.float32
    else:
        nomask = bool(cfg_nomask)
    eps = 1e-5

    device = inp.device
    B, N, _, _ = inp.shape
    M = B * N * N                       # total rows for row‑wise ops
    buffers = _rank01_buffers(device, B, N, dim, hidden_dim)

    # --------------------------------------------------------------
    # 1) Row‑wise LayerNorm (fp16)
    # --------------------------------------------------------------
    x_norm = _row_layernorm_fp16(
        inp,
        weights["norm.weight"],
        weights["norm.bias"],
        eps=eps,
        out=buffers["x_norm"],
    )  # (B, N, N, C) fp16

    # --------------------------------------------------------------
    # 2) Prepare projection / gate weights (C, H) in fp16, column‑major
    # --------------------------------------------------------------
    left_proj_w_T, right_proj_w_T, left_gate_w_T, right_gate_w_T, out_gate_w_T, to_out_w_T = (
        _rank01_weights_fp16(weights)
    )

    # --------------------------------------------------------------
    # 3) Mask handling (flattened) – optional
    # --------------------------------------------------------------
    if not nomask and mask is not None:
        mask_flat = mask.reshape(M).to(torch.float16).contiguous()
        MASKED = 1
    else:
        mask_flat = buffers["empty_mask"]
        MASKED = 0

    # --------------------------------------------------------------
    # 4) Allocate buffers for the fused projection / gating kernel
    # --------------------------------------------------------------
    left = buffers["left"]
    right = buffers["right"]
    out_gate = buffers["out_gate"]

    # --------------------------------------------------------------
    # 5) Fused projection + gating + (optional) mask
    # --------------------------------------------------------------
    BLOCK_M = 64      # rows per program (B·N·N)
    BLOCK_H = 64      # hidden‑dim block
    BLOCK_K = 32      # input‑channel tile size

    grid_proj = (triton.cdiv(M, BLOCK_M), triton.cdiv(hidden_dim, BLOCK_H))
    _proj_gate_mask_kernel[grid_proj](
        x_norm,
        mask_flat,
        left_proj_w_T,
        left_gate_w_T,
        right_proj_w_T,
        right_gate_w_T,
        out_gate_w_T,
        left,
        right,
        out_gate,
        M,
        N,
        dim,
        hidden_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H,
        BLOCK_K=BLOCK_K,
        MASKED=MASKED,
        num_warps=4,
    )

    # --------------------------------------------------------------
    # 6) Pairwise multiplication (batched GEMM)
    # --------------------------------------------------------------
    left_mat = left.view(B * hidden_dim, N, N)                 # (B*H, N, N)
    right_mat = right.view(B * hidden_dim, N, N).transpose(1, 2)  # (B*H, N, N)
    hidden_fp16 = buffers["hidden"]
    torch.bmm(left_mat, right_mat, out=hidden_fp16)             # (B*H, N, N) fp16
    hidden = hidden_fp16.view(B, hidden_dim, N, N)               # (B, H, N, N) fp16

    # --------------------------------------------------------------
    # 7) Fused hidden‑dim LN → out‑gate → final linear
    # --------------------------------------------------------------
    to_out_norm_w = weights["to_out_norm.weight"]   # (H,) fp32
    to_out_norm_b = weights["to_out_norm.bias"]    # (H,) fp32
    out = torch.empty((B, N, N, dim), dtype=torch.float32, device=device)

    BLOCK_M_OUT = 64
    BLOCK_H_OUT = 128   # covers the whole hidden dimension (H≤128)
    BLOCK_D_OUT = 64

    grid_out = (triton.cdiv(B * N * N, BLOCK_M_OUT),)

    _ln_gate_out_linear_fused_kernel[grid_out](
        hidden.view(-1),                     # flat fp16 hidden
        out_gate.view(-1),                   # flat fp16 out‑gate
        to_out_norm_w,
        to_out_norm_b,
        to_out_w_T,
        out,
        B,
        N,
        hidden_dim,
        dim,
        eps,
        BLOCK_M=BLOCK_M_OUT,
        BLOCK_H=BLOCK_H_OUT,
        BLOCK_D=BLOCK_D_OUT,
        num_warps=4,
    )

    return out


def _v48_c384_path(
    data: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict]
) -> torch.Tensor:
    inp, mask, weights, cfg = data
    dim = cfg["dim"]
    hidden_dim = cfg["hidden_dim"]
    cfg_nomask = cfg.get("nomask", None)
    if cfg_nomask is None:
        nomask = mask is None or mask.dtype == torch.float32
    else:
        nomask = bool(cfg_nomask)

    device = inp.device
    B, N, _, _ = inp.shape
    M = B * N * N
    ext = _get_ext()

    x_norm = ext.trimul_v48_layernorm(
        inp,
        weights["norm.weight"],
        weights["norm.bias"],
        int(hidden_dim),
    )

    left_proj_w_T, right_proj_w_T, left_gate_w_T, right_gate_w_T, out_gate_w_T = (
        _c384_proj_gate_weights_fp16(weights)
    )

    if not nomask and mask is not None:
        mask_flat = mask.reshape(M).to(torch.float16).contiguous()
        masked = 1
    else:
        mask_flat = torch.empty(0, dtype=torch.float16, device=device)
        masked = 0

    lr5 = _v48_lr5_buffer(device, B, N, dim, hidden_dim)

    block_m = int(os.environ.get("TRIMUL_V48_BLOCK_M", "64"))
    block_h = int(os.environ.get("TRIMUL_V48_BLOCK_H", "64"))
    block_k = int(os.environ.get("TRIMUL_V48_BLOCK_K", "32"))
    num_warps = int(os.environ.get("TRIMUL_V48_WARPS", "4"))
    grid_proj = (triton.cdiv(M, block_m), triton.cdiv(hidden_dim, block_h))
    _proj_gate_mask_hmajor_kernel[grid_proj](
        x_norm,
        mask_flat,
        left_proj_w_T,
        left_gate_w_T,
        right_proj_w_T,
        right_gate_w_T,
        out_gate_w_T,
        lr5,
        M,
        dim,
        hidden_dim,
        BLOCK_M=block_m,
        BLOCK_H=block_h,
        BLOCK_K=block_k,
        MASKED=masked,
        num_warps=num_warps,
    )

    out = ext.trimul_v48_tail(
        lr5,
        weights["to_out_norm.weight"],
        weights["to_out_norm.bias"],
        weights["to_out.weight"],
        0,
        int(_USE_OLD_HIDDEN),
    )

    return out


@torch.no_grad()
def custom_kernel(data: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]) -> torch.Tensor:
    x, mask, _, cfg = data
    B, N, _, C = x.shape
    hidden_dim = int(cfg["hidden_dim"])
    cfg_nomask = cfg.get("nomask", None)
    if cfg_nomask is None:
        is_nomask = mask is None or mask.dtype == torch.float32
    else:
        is_nomask = bool(cfg_nomask)
    if hidden_dim == 128 and C == 128 and B == 1 and N in (512, 768, 1024):
        return _rank01_c128_path(data)
    if (
        hidden_dim == 128
        and C == 384
        and B == 1
        and (N == 768 or (N == 1024 and is_nomask))
        and os.environ.get("TRIMUL_V48_DISABLE_C384", "") != "1"
    ):
        return _v48_c384_path(data)
    return _v40_path(data)
