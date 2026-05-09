from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import torch
from torch.utils.cpp_extension import load_inline

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

torch::Tensor trimul_v30_forward(
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
    torch::Tensor to_out_weight
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("trimul_v30_forward", &trimul_v30_forward, "trimul v30 cuda hidden tile dispatch");
}
"""

    cuda_src = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDABlas.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <unordered_map>

namespace {

constexpr float kEps = 1.0e-5f;

static void cublas_check(cublasStatus_t status, const char* msg) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, msg);
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

struct DeviceCaches {
  WorkspaceCache workspace;
  PackedWeightsCache packed_weights;
};

static DeviceCaches& get_device_caches(int device) {
  thread_local std::unordered_map<int, DeviceCaches> per_device;
  return per_device[device];
}

static inline int64_t tensor_version(const at::Tensor& t) {
  return static_cast<int64_t>(t._version());
}

}  // namespace

torch::Tensor trimul_v30_forward(
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
    torch::Tensor to_out_weight) {
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
  } else if (dim == 384 && rows > static_cast<int64_t>(2) * 256 * 256) {
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
  PackedWeightsCache& packed_cache = caches.packed_weights;
  const void* p0 = left_proj_weight.data_ptr<float>();
  const void* p1 = right_proj_weight.data_ptr<float>();
  const void* p2 = left_gate_weight.data_ptr<float>();
  const void* p3 = right_gate_weight.data_ptr<float>();
  const void* p4 = out_gate_weight.data_ptr<float>();
  const void* p5 = to_out_weight.data_ptr<float>();
  const int64_t v0 = tensor_version(left_proj_weight);
  const int64_t v1 = tensor_version(right_proj_weight);
  const int64_t v2 = tensor_version(left_gate_weight);
  const int64_t v3 = tensor_version(right_gate_weight);
  const int64_t v4 = tensor_version(out_gate_weight);
  const int64_t v5 = tensor_version(to_out_weight);
  const bool packed_hit =
      packed_cache.packed.defined() &&
      packed_cache.seg_elems == seg_elems &&
      packed_cache.p0 == p0 &&
      packed_cache.p1 == p1 &&
      packed_cache.p2 == p2 &&
      packed_cache.p3 == p3 &&
      packed_cache.p4 == p4 &&
      packed_cache.p5 == p5 &&
      packed_cache.v0 == v0 &&
      packed_cache.v1 == v1 &&
      packed_cache.v2 == v2 &&
      packed_cache.v3 == v3 &&
      packed_cache.v4 == v4 &&
      packed_cache.v5 == v5;
  if (!packed_hit) {
    packed_cache.packed = torch::empty({6 * seg_elems}, opts_f16);
    const int pack_threads = 256;
    const int pack_blocks = static_cast<int>((6 * seg_elems + pack_threads - 1) / pack_threads);
    pack6_f32_to_f16_kernel<<<pack_blocks, pack_threads>>>(
        left_proj_weight.data_ptr<float>(),
        right_proj_weight.data_ptr<float>(),
        left_gate_weight.data_ptr<float>(),
        right_gate_weight.data_ptr<float>(),
        out_gate_weight.data_ptr<float>(),
        to_out_weight.data_ptr<float>(),
        reinterpret_cast<__half*>(packed_cache.packed.data_ptr<at::Half>()),
        seg_elems);
    packed_cache.p0 = p0;
    packed_cache.p1 = p1;
    packed_cache.p2 = p2;
    packed_cache.p3 = p3;
    packed_cache.p4 = p4;
    packed_cache.p5 = p5;
    packed_cache.v0 = v0;
    packed_cache.v1 = v1;
    packed_cache.v2 = v2;
    packed_cache.v3 = v3;
    packed_cache.v4 = v4;
    packed_cache.v5 = v5;
    packed_cache.seg_elems = seg_elems;
  }

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  const __half* packed_ptr = reinterpret_cast<const __half*>(packed_cache.packed.data_ptr<at::Half>());
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
  if (mask.scalar_type() == torch::kFloat32) {
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
  if (hidden_dim == 128) {
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
"""

    _EXT = load_inline(
        name="trimul_v30_cuda_ext_hidden_tile_dispatch",
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
def custom_kernel(data: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]) -> torch.Tensor:
    x, mask, weights, _ = data
    ext = _get_ext()
    return ext.trimul_v30_forward(
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
    )
