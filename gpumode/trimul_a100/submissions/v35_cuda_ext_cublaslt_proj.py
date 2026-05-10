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

torch::Tensor trimul_v35_forward(
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
  m.def("trimul_v35_forward", &trimul_v35_forward, "trimul v35 cuda cublaslt proj/out");
}
"""

    cuda_src = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdint>
#include <unordered_map>

namespace {

constexpr float kEps = 1.0e-5f;

static void cublas_check(cublasStatus_t status, const char* msg) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, msg);
}

static bool env_flag(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static bool env_present(const char* name) {
  return std::getenv(name) != nullptr;
}

static int env_int(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value) {
    return fallback;
  }
  return static_cast<int>(parsed);
}

struct RuntimeConfig {
  bool has_lt_override = false;
  bool force_lt_proj = false;
  bool force_lt_out = false;
  bool disable_lt_auto = false;
  int lt_proj_index = 0;
  int lt_out_index = 0;
  int lt_workspace_mb = 32;
};

static RuntimeConfig make_runtime_config() {
  RuntimeConfig cfg;
  cfg.has_lt_override =
      env_present("TRIMUL_USE_LT") ||
      env_present("TRIMUL_USE_LT_PROJ") ||
      env_present("TRIMUL_USE_LT_OUT");
  if (cfg.has_lt_override) {
    const bool use_lt_all = env_flag("TRIMUL_USE_LT");
    cfg.force_lt_proj = use_lt_all || env_flag("TRIMUL_USE_LT_PROJ");
    cfg.force_lt_out = use_lt_all || env_flag("TRIMUL_USE_LT_OUT");
  }
  cfg.disable_lt_auto = env_flag("TRIMUL_DISABLE_LT_AUTO");
  cfg.lt_proj_index = env_int("TRIMUL_LT_PROJ_INDEX", 0);
  cfg.lt_out_index = env_int("TRIMUL_LT_OUT_INDEX", 0);
  cfg.lt_workspace_mb = env_int("TRIMUL_LT_WORKSPACE_MB", 32);
  if (cfg.lt_workspace_mb < 0) {
    cfg.lt_workspace_mb = 0;
  }
  return cfg;
}

static const RuntimeConfig kRuntimeConfig = make_runtime_config();

static const RuntimeConfig& runtime_config() {
  return kRuntimeConfig;
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

struct LtGemmPlan {
  int64_t m = -1;
  int64_t k = -1;
  int64_t n = -1;
  size_t workspace_bytes = 0;
  int heuristic_index = -1;
  bool ready = false;
  bool disabled = false;
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulAlgo_t algo;
};

static void lt_plan_clear(LtGemmPlan& plan) {
  if (plan.op_desc != nullptr) {
    cublasLtMatmulDescDestroy(plan.op_desc);
  }
  if (plan.a_desc != nullptr) {
    cublasLtMatrixLayoutDestroy(plan.a_desc);
  }
  if (plan.b_desc != nullptr) {
    cublasLtMatrixLayoutDestroy(plan.b_desc);
  }
  if (plan.c_desc != nullptr) {
    cublasLtMatrixLayoutDestroy(plan.c_desc);
  }
  plan.m = -1;
  plan.k = -1;
  plan.n = -1;
  plan.workspace_bytes = 0;
  plan.heuristic_index = -1;
  plan.ready = false;
  plan.disabled = false;
  plan.op_desc = nullptr;
  plan.a_desc = nullptr;
  plan.b_desc = nullptr;
  plan.c_desc = nullptr;
}

static bool lt_prepare_abt(
    cublasLtHandle_t lt_handle,
    LtGemmPlan& plan,
    int64_t m,
    int64_t k,
    int64_t n,
    size_t workspace_bytes,
    int heuristic_index) {
  if (plan.ready &&
      plan.m == m &&
      plan.k == k &&
      plan.n == n &&
      plan.workspace_bytes == workspace_bytes &&
      plan.heuristic_index == heuristic_index) {
    return !plan.disabled;
  }

  lt_plan_clear(plan);
  plan.m = m;
  plan.k = k;
  plan.n = n;
  plan.workspace_bytes = workspace_bytes;
  plan.heuristic_index = heuristic_index;

  cublasStatus_t status = cublasLtMatmulDescCreate(
      &plan.op_desc,
      CUBLAS_COMPUTE_32F_FAST_16F,
      CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) {
    lt_plan_clear(plan);
    return false;
  }

  const cublasOperation_t transa = CUBLAS_OP_T;
  const cublasOperation_t transb = CUBLAS_OP_N;
  status = cublasLtMatmulDescSetAttribute(
      plan.op_desc,
      CUBLASLT_MATMUL_DESC_TRANSA,
      &transa,
      sizeof(transa));
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulDescSetAttribute(
        plan.op_desc,
        CUBLASLT_MATMUL_DESC_TRANSB,
        &transb,
        sizeof(transb));
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatrixLayoutCreate(
        &plan.a_desc,
        CUDA_R_16F,
        static_cast<uint64_t>(k),
        static_cast<uint64_t>(n),
        static_cast<int64_t>(k));
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatrixLayoutCreate(
        &plan.b_desc,
        CUDA_R_16F,
        static_cast<uint64_t>(k),
        static_cast<uint64_t>(m),
        static_cast<int64_t>(k));
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatrixLayoutCreate(
        &plan.c_desc,
        CUDA_R_16F,
        static_cast<uint64_t>(n),
        static_cast<uint64_t>(m),
        static_cast<int64_t>(n));
  }
  if (status != CUBLAS_STATUS_SUCCESS) {
    lt_plan_clear(plan);
    return false;
  }

  cublasLtMatmulPreference_t preference = nullptr;
  status = cublasLtMatmulPreferenceCreate(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) {
    lt_plan_clear(plan);
    return false;
  }
  status = cublasLtMatmulPreferenceSetAttribute(
      preference,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspace_bytes,
      sizeof(workspace_bytes));
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulPreferenceDestroy(preference);
    lt_plan_clear(plan);
    return false;
  }

  constexpr int kRequestedAlgos = 16;
  cublasLtMatmulHeuristicResult_t results[kRequestedAlgos];
  int returned = 0;
  status = cublasLtMatmulAlgoGetHeuristic(
      lt_handle,
      plan.op_desc,
      plan.a_desc,
      plan.b_desc,
      plan.c_desc,
      plan.c_desc,
      preference,
      kRequestedAlgos,
      results,
      &returned);
  cublasLtMatmulPreferenceDestroy(preference);
  if (status != CUBLAS_STATUS_SUCCESS || returned <= 0) {
    lt_plan_clear(plan);
    return false;
  }

  int selected = heuristic_index;
  if (selected < 0) {
    selected = 0;
  }
  if (selected >= returned) {
    selected = returned - 1;
  }
  plan.algo = results[selected].algo;
  plan.ready = true;
  plan.disabled = false;
  return true;
}

static bool gemm_f16_abt_lt(
    cublasLtHandle_t lt_handle,
    LtGemmPlan& plan,
    const __half* a_rm,
    const __half* b_rm,
    __half* c_rm,
    int64_t m,
    int64_t k,
    int64_t n,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    int heuristic_index) {
  if (!lt_prepare_abt(lt_handle, plan, m, k, n, workspace_bytes, heuristic_index)) {
    return false;
  }
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const cublasStatus_t status = cublasLtMatmul(
      lt_handle,
      plan.op_desc,
      &alpha,
      b_rm,
      plan.a_desc,
      a_rm,
      plan.b_desc,
      &beta,
      c_rm,
      plan.c_desc,
      c_rm,
      plan.c_desc,
      &plan.algo,
      workspace,
      workspace_bytes,
      stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    plan.disabled = true;
    return false;
  }
  return true;
}

struct WorkspaceCache {
  at::Tensor xhat;
  at::Tensor lr5;
  at::Tensor central;
  at::Tensor hidden_rows;
  at::Tensor lt_workspace;
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
  LtGemmPlan proj_lt;
  LtGemmPlan out_lt;
  cublasLtHandle_t lt_handle = nullptr;
};

static DeviceCaches& get_device_caches(int device) {
  thread_local std::unordered_map<int, DeviceCaches> per_device;
  return per_device[device];
}

static inline int64_t tensor_version(const at::Tensor& t) {
  return static_cast<int64_t>(t._version());
}

}  // namespace

torch::Tensor trimul_v35_forward(
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

  const RuntimeConfig& rt = runtime_config();
  bool use_lt_proj = false;
  bool use_lt_out = false;
  int lt_proj_index = rt.lt_proj_index;
  int lt_out_index = rt.lt_out_index;
  if (rt.has_lt_override) {
    use_lt_proj = rt.force_lt_proj;
    use_lt_out = rt.force_lt_out;
  } else if (!rt.disable_lt_auto && hidden_dim == 128 && bs == 1) {
    lt_proj_index = 0;
    lt_out_index = 0;
    if (dim == 128 && n == 512) {
      use_lt_proj = true;
      use_lt_out = true;
    } else if (dim == 128 && (n == 768 || n == 1024)) {
      use_lt_proj = true;
      use_lt_out = true;
      lt_out_index = 1;
    } else if (dim == 384 && (n == 768 || n == 1024)) {
      use_lt_proj = true;
      use_lt_out = true;
      lt_proj_index = 1;
    }
  }
  const size_t lt_workspace_bytes =
      static_cast<size_t>(rt.lt_workspace_mb) * static_cast<size_t>(1024 * 1024);
  void* lt_workspace_ptr = nullptr;
  if ((use_lt_proj || use_lt_out) && lt_workspace_bytes > 0) {
    if (!workspace.lt_workspace.defined() ||
        workspace.lt_workspace.numel() < static_cast<int64_t>(lt_workspace_bytes)) {
      auto opts_u8 = x.options().dtype(torch::kUInt8);
      workspace.lt_workspace = torch::empty({static_cast<int64_t>(lt_workspace_bytes)}, opts_u8);
    }
    lt_workspace_ptr = workspace.lt_workspace.data_ptr<uint8_t>();
  }
  cublasLtHandle_t lt_handle = nullptr;
  cudaStream_t stream = nullptr;
  if (use_lt_proj || use_lt_out) {
    if (caches.lt_handle == nullptr) {
      cublas_check(cublasLtCreate(&caches.lt_handle), "cublasLtCreate failed");
    }
    lt_handle = caches.lt_handle;
    stream = at::cuda::getCurrentCUDAStream();
  }

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
  bool lt_proj_ok = false;
  if (use_lt_proj) {
    lt_proj_ok = gemm_f16_abt_lt(
        lt_handle,
        caches.proj_lt,
        packed_ptr,
        reinterpret_cast<const __half*>(xhat.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(lr5.data_ptr<at::Half>()),
        5 * hidden_dim,
        dim,
        rows,
        lt_workspace_ptr,
        lt_workspace_bytes,
        stream,
        lt_proj_index);
  }
  if (!lt_proj_ok) {
    gemm_f16_abt(
        handle,
        packed_ptr,
        reinterpret_cast<const __half*>(xhat.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(lr5.data_ptr<at::Half>()),
        5 * hidden_dim,
        dim,
        rows);
  }

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
  bool lt_out_ok = false;
  if (use_lt_out) {
    lt_out_ok = gemm_f16_abt_lt(
        lt_handle,
        caches.out_lt,
        reinterpret_cast<const __half*>(hidden_rows.data_ptr<at::Half>()),
        to_out_ptr,
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        rows,
        hidden_dim,
        dim,
        lt_workspace_ptr,
        lt_workspace_bytes,
        stream,
        lt_out_index);
  }
  if (!lt_out_ok) {
    gemm_f16_abt(
        handle,
        reinterpret_cast<const __half*>(hidden_rows.data_ptr<at::Half>()),
        to_out_ptr,
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        rows,
        hidden_dim,
        dim);
  }

  return y;
}
"""

    _EXT = load_inline(
        name="trimul_v35_cuda_ext_cublaslt_proj",
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
        extra_ldflags=["-lcublasLt"],
        verbose=False,
    )
    return _EXT


@torch.no_grad()
def custom_kernel(data: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]) -> torch.Tensor:
    x, mask, weights, _ = data
    ext = _get_ext()
    return ext.trimul_v35_forward(
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
