#!/usr/bin/env python3
import argparse
import os
import statistics
import textwrap

import torch
from torch.utils.cpp_extension import load_inline


CPP_SRC = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <limits>
#include <sstream>
#include <vector>

#define CHECK_CUBLAS(call)                                                        \
  do {                                                                            \
    cublasStatus_t status = (call);                                               \
    if (status != CUBLAS_STATUS_SUCCESS) {                                        \
      std::ostringstream oss;                                                     \
      oss << "cuBLAS error " << static_cast<int>(status) << " at " << __LINE__; \
      throw std::runtime_error(oss.str());                                        \
    }                                                                             \
  } while (0)

#define CHECK_CUDA_RUNTIME(call) C10_CUDA_CHECK(call)

struct LtDescriptors {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr;
  cublasLtMatrixLayout_t b = nullptr;
  cublasLtMatrixLayout_t c = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;

  ~LtDescriptors() {
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (op) cublasLtMatmulDescDestroy(op);
  }
};

static LtDescriptors make_descriptors(
    int64_t m, int64_t n, int64_t k, size_t workspace_bytes) {
  LtDescriptors desc;
  cublasOperation_t trans = CUBLAS_OP_N;
  cublasLtOrder_t order = CUBLASLT_ORDER_ROW;

  CHECK_CUBLAS(cublasLtMatmulDescCreate(&desc.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      desc.op, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      desc.op, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)));

  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&desc.a, CUDA_R_16F, m, k, k));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&desc.b, CUDA_R_16F, k, n, n));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&desc.c, CUDA_R_16F, m, n, n));
  CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
      desc.a, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
  CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
      desc.b, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
  CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
      desc.c, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

  CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&desc.pref));
  CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      desc.pref,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspace_bytes,
      sizeof(workspace_bytes)));
  return desc;
}

std::vector<std::vector<double>> lt_benchmark(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    int64_t workspace_bytes,
    int requested_algos,
    int warmup,
    int iters) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(), "all tensors must be CUDA");
  TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be fp16");
  TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be fp16");
  TORCH_CHECK(C.dtype() == torch::kFloat16, "C must be fp16");
  TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous(), "tensors must be contiguous");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2, "expected 2D tensors");
  TORCH_CHECK(A.size(1) == B.size(0), "K mismatch");
  TORCH_CHECK(C.size(0) == A.size(0) && C.size(1) == B.size(1), "C shape mismatch");

  const int64_t m = A.size(0);
  const int64_t k = A.size(1);
  const int64_t n = B.size(1);
  const float alpha = 1.0f;
  const float beta = 0.0f;

  cublasLtHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasLtCreate(&handle));
  auto handle_guard = std::unique_ptr<std::remove_pointer<cublasLtHandle_t>::type, decltype(&cublasLtDestroy)>(
      handle, &cublasLtDestroy);

  auto desc = make_descriptors(m, n, k, static_cast<size_t>(workspace_bytes));
  std::vector<cublasLtMatmulHeuristicResult_t> heuristic(requested_algos);
  int returned = 0;
  CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
      handle,
      desc.op,
      desc.a,
      desc.b,
      desc.c,
      desc.c,
      desc.pref,
      requested_algos,
      heuristic.data(),
      &returned));

  torch::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_bytes > 0) {
    workspace = torch::empty({workspace_bytes}, C.options().dtype(torch::kUInt8));
    workspace_ptr = workspace.data_ptr();
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  std::vector<std::vector<double>> rows;

  for (int algo_idx = 0; algo_idx < returned; ++algo_idx) {
    auto& result = heuristic[algo_idx];
    if (result.state != CUBLAS_STATUS_SUCCESS) {
      rows.push_back({
          static_cast<double>(algo_idx),
          static_cast<double>(result.state),
          static_cast<double>(result.workspaceSize),
          static_cast<double>(result.wavesCount),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
      });
      continue;
    }

    for (int i = 0; i < warmup; ++i) {
      CHECK_CUBLAS(cublasLtMatmul(
          handle,
          desc.op,
          &alpha,
          A.data_ptr(),
          desc.a,
          B.data_ptr(),
          desc.b,
          &beta,
          C.data_ptr(),
          desc.c,
          C.data_ptr(),
          desc.c,
          &result.algo,
          workspace_ptr,
          static_cast<size_t>(workspace_bytes),
          stream));
    }
    CHECK_CUDA_RUNTIME(cudaStreamSynchronize(stream));

    float total_ms = 0.0f;
    float best_ms = std::numeric_limits<float>::infinity();
    float worst_ms = 0.0f;
    for (int i = 0; i < iters; ++i) {
      cudaEvent_t start, stop;
      CHECK_CUDA_RUNTIME(cudaEventCreate(&start));
      CHECK_CUDA_RUNTIME(cudaEventCreate(&stop));
      CHECK_CUDA_RUNTIME(cudaEventRecord(start, stream));
      CHECK_CUBLAS(cublasLtMatmul(
          handle,
          desc.op,
          &alpha,
          A.data_ptr(),
          desc.a,
          B.data_ptr(),
          desc.b,
          &beta,
          C.data_ptr(),
          desc.c,
          C.data_ptr(),
          desc.c,
          &result.algo,
          workspace_ptr,
          static_cast<size_t>(workspace_bytes),
          stream));
      CHECK_CUDA_RUNTIME(cudaEventRecord(stop, stream));
      CHECK_CUDA_RUNTIME(cudaEventSynchronize(stop));
      float elapsed_ms = 0.0f;
      CHECK_CUDA_RUNTIME(cudaEventElapsedTime(&elapsed_ms, start, stop));
      CHECK_CUDA_RUNTIME(cudaEventDestroy(start));
      CHECK_CUDA_RUNTIME(cudaEventDestroy(stop));
      total_ms += elapsed_ms;
      best_ms = std::min(best_ms, elapsed_ms);
      worst_ms = std::max(worst_ms, elapsed_ms);
    }

    rows.push_back({
        static_cast<double>(algo_idx),
        static_cast<double>(result.state),
        static_cast<double>(result.workspaceSize),
        static_cast<double>(result.wavesCount),
        static_cast<double>(total_ms * 1000.0f / iters),
        static_cast<double>(best_ms * 1000.0f),
        static_cast<double>(worst_ms * 1000.0f),
    });
  }
  return rows;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lt_benchmark", &lt_benchmark, "Benchmark cuBLASLt heuristic candidates");
}
"""


def build_extension():
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    local_bin = os.path.expanduser("~/.local/bin")
    os.environ["PATH"] = local_bin + os.pathsep + os.environ.get("PATH", "")
    return load_inline(
        name="a100_cublaslt_probe_ext",
        cpp_sources=[CPP_SRC],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=["-lcublasLt", "-lcublas"],
        verbose=False,
        with_cuda=True,
    )


def parse_workspace_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def time_torch_mm(a, b, c, warmup, iters):
    for _ in range(warmup):
        torch.mm(a, b, out=c)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.mm(a, b, out=c)
        stop.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(stop) * 1000.0)
    return statistics.fmean(times), min(times), max(times)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            Probe cuBLASLt heuristic candidates for GPUMODE matmul_v2 A100.
            The shape is fixed to the final benchmark by default because it
            dominates both the official score and the rank-1 gap.
            """
        ),
    )
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=5120)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--workspaces", default="0,1048576,8388608,33554432")
    parser.add_argument("--algos", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123456)
    args = parser.parse_args()

    ext = build_extension()
    gen = torch.Generator(device="cuda")
    gen.manual_seed(args.seed)
    a = torch.randn((args.m, args.k), device="cuda", dtype=torch.float16, generator=gen)
    b = torch.randn((args.k, args.n), device="cuda", dtype=torch.float16, generator=gen)
    c = torch.empty((args.m, args.n), device="cuda", dtype=torch.float16)
    ref = torch.empty_like(c)

    torch_mean, torch_best, torch_worst = time_torch_mm(a, b, ref, args.warmup, args.iters)
    print(f"device,{torch.cuda.get_device_name()}")
    print(f"torch,{torch.__version__}")
    print(f"shape,{args.m},{args.n},{args.k}")
    print(f"torch_mm,{torch_mean:.3f},{torch_best:.3f},{torch_worst:.3f}")
    print("workspace_bytes,algo_idx,state,heur_workspace,waves,mean_us,best_us,worst_us,max_diff")

    for workspace in parse_workspace_list(args.workspaces):
        rows = ext.lt_benchmark(a, b, c, workspace, args.algos, args.warmup, args.iters)
        torch.cuda.synchronize()
        max_diff = (c - ref).abs().max().item()
        for row in rows:
            algo_idx, state, heur_ws, waves, mean_us, best_us, worst_us = row
            print(
                f"{workspace},{int(algo_idx)},{int(state)},{int(heur_ws)},"
                f"{waves:.3f},{mean_us:.3f},{best_us:.3f},{worst_us:.3f},{max_diff}"
            )


if __name__ == "__main__":
    main()
