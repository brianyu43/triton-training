#!POPCORN leaderboard matmul_v2
#!POPCORN gpus A100

import os
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t


CPP_SRC = r"""
#include <cublasLt.h>
#include <torch/extension.h>

#include <stdexcept>
#include <vector>

#define CHECK_CUBLAS(call)                         \
  do {                                             \
    cublasStatus_t status = (call);                \
    if (status != CUBLAS_STATUS_SUCCESS) {         \
      throw std::runtime_error("cuBLAS error");   \
    }                                              \
  } while (0)


struct LtPlan {
  bool initialized = false;
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulAlgo_t algo;
  torch::Tensor workspace;
  size_t workspace_bytes = 33554432;

  void init(torch::Tensor C) {
    if (initialized) return;

    const int64_t m = 4096;
    const int64_t n = 5120;
    const int64_t k = 4096;
    const int requested = 8;
    const int wanted_idx = 2;

    cublasOperation_t trans = CUBLAS_OP_N;
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;

    CHECK_CUBLAS(cublasLtCreate(&handle));
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        op, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        op, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)));

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16F, m, k, k));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16F, k, n, n));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16F, m, n, n));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

    cublasLtMatmulPreference_t pref = nullptr;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_bytes,
        sizeof(workspace_bytes)));

    std::vector<cublasLtMatmulHeuristicResult_t> heuristic(requested);
    int returned = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        handle,
        op,
        a_desc,
        b_desc,
        c_desc,
        c_desc,
        pref,
        requested,
        heuristic.data(),
        &returned));
    CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(pref));
    if (returned <= 0) {
      throw std::runtime_error("cuBLASLt returned no heuristic algorithms");
    }
    int chosen = wanted_idx < returned ? wanted_idx : 0;
    algo = heuristic[chosen].algo;
    workspace = torch::empty({static_cast<int64_t>(workspace_bytes)}, C.options().dtype(torch::kUInt8));
    initialized = true;
  }
};

static LtPlan plan;

void cublaslt_matmul(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(), "expected CUDA tensors");
  TORCH_CHECK(A.dtype() == torch::kFloat16 && B.dtype() == torch::kFloat16 && C.dtype() == torch::kFloat16,
              "expected fp16 tensors");
  TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous(), "expected contiguous tensors");
  plan.init(C);

  const float alpha = 1.0f;
  const float beta = 0.0f;
  CHECK_CUBLAS(cublasLtMatmul(
      plan.handle,
      plan.op,
      &alpha,
      A.data_ptr(),
      plan.a_desc,
      B.data_ptr(),
      plan.b_desc,
      &beta,
      C.data_ptr(),
      plan.c_desc,
      C.data_ptr(),
      plan.c_desc,
      &plan.algo,
      plan.workspace.data_ptr(),
      plan.workspace_bytes,
      0));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cublaslt_matmul", &cublaslt_matmul, "fixed-shape cuBLASLt matmul");
}
"""


os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
_local_bin = os.path.expanduser("~/.local/bin")
os.environ["PATH"] = _local_bin + os.pathsep + os.environ.get("PATH", "")

_EXT = load_inline(
    name="a100_v2_cublaslt_ws32m_idx2_s0_ext",
    cpp_sources=[CPP_SRC],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    extra_ldflags=["-lcublasLt", "-lcublas"],
    verbose=False,
    with_cuda=True,
)


def custom_kernel(data: input_t) -> output_t:
    a, b, c = data
    if a.shape == (4096, 4096) and b.shape == (4096, 5120):
        _EXT.cublaslt_matmul(a, b, c)
    else:
        torch.mm(a, b, out=c)
    return c
