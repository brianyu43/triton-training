from __future__ import annotations

import argparse
import os
import subprocess
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
    checksum: float | None = None
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


def build_ext(verbose_build: bool):
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    os.environ["MAX_JOBS"] = "4"

    cpp_src = r"""
#include <torch/extension.h>

void empty_launch(torch::Tensor sink);
void load_scalar_sink(torch::Tensor x, torch::Tensor sink, int blocks, int threads);
void load_float2_sink(torch::Tensor x, torch::Tensor sink, int blocks, int threads);
void load_float4_sink(torch::Tensor x, torch::Tensor sink, int blocks, int threads);
void load_float4_sink_var(torch::Tensor x, torch::Tensor sink, int blocks, int threads);
void load_float4_asm_sink(torch::Tensor x, int blocks, int threads);
void load_float4_block_chunk_sink(torch::Tensor x, torch::Tensor sink, int blocks, int threads);
void load_float4_ldg_sink(torch::Tensor x, torch::Tensor sink, int blocks, int threads);
void load_float4_cg_sink(torch::Tensor x, torch::Tensor sink, int blocks, int threads);
void load_float4_ca_sink(torch::Tensor x, torch::Tensor sink, int blocks, int threads);
void load_float4_tile4_sink(torch::Tensor x, torch::Tensor sink, int blocks, int threads);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("empty_launch", &empty_launch, "empty launch");
  m.def("load_scalar_sink", &load_scalar_sink, "scalar full-read sink");
  m.def("load_float2_sink", &load_float2_sink, "float2 full-read sink");
  m.def("load_float4_sink", &load_float4_sink, "float4 full-read sink");
  m.def("load_float4_sink_var", &load_float4_sink_var, "float4 full-read sink, variable thread launch");
  m.def("load_float4_asm_sink", &load_float4_asm_sink, "float4 full-read asm sink");
  m.def("load_float4_block_chunk_sink", &load_float4_block_chunk_sink, "float4 block-contiguous chunk sink");
  m.def("load_float4_ldg_sink", &load_float4_ldg_sink, "__ldg float4 full-read sink");
  m.def("load_float4_cg_sink", &load_float4_cg_sink, "ld.global.cg float4 full-read sink");
  m.def("load_float4_ca_sink", &load_float4_ca_sink, "ld.global.ca float4 full-read sink");
  m.def("load_float4_tile4_sink", &load_float4_tile4_sink, "float4 tile4 full-read sink");
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

__device__ __forceinline__ float sum2(float2 v) {
  return v.x + v.y;
}

__device__ __forceinline__ float sum4(float4 v) {
  return (v.x + v.y) + (v.z + v.w);
}

__device__ __forceinline__ float4 load_cg_v4(const float* ptr) {
  float4 v;
  asm volatile(
      "ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];"
      : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
      : "l"(ptr));
  return v;
}

__device__ __forceinline__ float4 load_ca_v4(const float* ptr) {
  float4 v;
  asm volatile(
      "ld.global.ca.v4.f32 {%0, %1, %2, %3}, [%4];"
      : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
      : "l"(ptr));
  return v;
}

__global__ void empty_kernel(float* __restrict__ sink) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    sink[0] = sink[0];
  }
}

__global__ __launch_bounds__(256, 8) void load_scalar_sink_kernel(
    const float* __restrict__ x,
    float* __restrict__ sink,
    int n) {
  const int stride = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;
  for (int idx = tid; idx < n; idx += stride) {
    sum += x[idx];
  }
  sink[tid] = sum;
}

__global__ __launch_bounds__(256, 8) void load_float2_sink_kernel(
    const float2* __restrict__ x2,
    float* __restrict__ sink,
    int n2) {
  const int stride = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;
  for (int idx = tid; idx < n2; idx += stride) {
    sum += sum2(x2[idx]);
  }
  sink[tid] = sum;
}

__global__ __launch_bounds__(256, 8) void load_float4_sink_kernel(
    const float4* __restrict__ x4,
    float* __restrict__ sink,
    int n4) {
  const int stride = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;
  for (int idx = tid; idx < n4; idx += stride) {
    sum += sum4(x4[idx]);
  }
  sink[tid] = sum;
}

__global__ void load_float4_sink_var_kernel(
    const float4* __restrict__ x4,
    float* __restrict__ sink,
    int n4) {
  const int stride = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;
  for (int idx = tid; idx < n4; idx += stride) {
    sum += sum4(x4[idx]);
  }
  sink[tid] = sum;
}

__global__ void load_float4_asm_sink_kernel(
    const float4* __restrict__ x4,
    int n4) {
  const int stride = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;
  for (int idx = tid; idx < n4; idx += stride) {
    sum += sum4(x4[idx]);
  }
  asm volatile("mov.f32 %0, %0;" : "+f"(sum));
}

__global__ void load_float4_block_chunk_sink_kernel(
    const float4* __restrict__ x4,
    float* __restrict__ sink,
    int n4) {
  const int chunk = (n4 + gridDim.x - 1) / gridDim.x;
  const int begin = blockIdx.x * chunk;
  const int end = min(begin + chunk, n4);
  float sum = 0.0f;
  for (int idx = begin + threadIdx.x; idx < end; idx += blockDim.x) {
    sum += sum4(x4[idx]);
  }
  sink[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

__global__ __launch_bounds__(256, 8) void load_float4_ldg_sink_kernel(
    const float4* __restrict__ x4,
    float* __restrict__ sink,
    int n4) {
  const int stride = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;
  for (int idx = tid; idx < n4; idx += stride) {
    sum += sum4(__ldg(x4 + idx));
  }
  sink[tid] = sum;
}

__global__ __launch_bounds__(256, 8) void load_float4_cg_sink_kernel(
    const float* __restrict__ x,
    float* __restrict__ sink,
    int n4) {
  const int stride = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;
  for (int idx = tid; idx < n4; idx += stride) {
    sum += sum4(load_cg_v4(x + 4 * idx));
  }
  sink[tid] = sum;
}

__global__ __launch_bounds__(256, 8) void load_float4_ca_sink_kernel(
    const float* __restrict__ x,
    float* __restrict__ sink,
    int n4) {
  const int stride = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;
  for (int idx = tid; idx < n4; idx += stride) {
    sum += sum4(load_ca_v4(x + 4 * idx));
  }
  sink[tid] = sum;
}

__global__ __launch_bounds__(256, 8) void load_float4_tile4_sink_kernel(
    const float4* __restrict__ x4,
    float* __restrict__ sink,
    int n4) {
  const int tile = blockDim.x * kTileVecsPerThread;
  const int grid_tile = gridDim.x * tile;
  int base = blockIdx.x * tile + threadIdx.x;

  float s0 = 0.0f;
  float s1 = 0.0f;
  float s2 = 0.0f;
  float s3 = 0.0f;

  for (; base < n4; base += grid_tile) {
    const int i0 = base;
    const int i1 = base + blockDim.x;
    const int i2 = base + 2 * blockDim.x;
    const int i3 = base + 3 * blockDim.x;
    if (i0 < n4) s0 += sum4(x4[i0]);
    if (i1 < n4) s1 += sum4(x4[i1]);
    if (i2 < n4) s2 += sum4(x4[i2]);
    if (i3 < n4) s3 += sum4(x4[i3]);
  }

  sink[blockIdx.x * blockDim.x + threadIdx.x] = (s0 + s1) + (s2 + s3);
}

}  // namespace

void empty_launch(torch::Tensor sink) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  empty_kernel<<<1, 1, 0, stream>>>(sink.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void load_scalar_sink(torch::Tensor x, torch::Tensor sink, int blocks, int threads) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(x.numel() <= static_cast<int64_t>(2147483647), "n must fit int32");
  load_scalar_sink_kernel<<<blocks, threads, 0, stream>>>(
      x.data_ptr<float>(), sink.data_ptr<float>(), static_cast<int>(x.numel()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void load_float2_sink(torch::Tensor x, torch::Tensor sink, int blocks, int threads) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK((x.numel() & 1) == 0, "n must be divisible by 2");
  const int64_t n2 = x.numel() >> 1;
  TORCH_CHECK(n2 <= static_cast<int64_t>(2147483647), "n/2 must fit int32");
  load_float2_sink_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const float2*>(x.data_ptr<float>()),
      sink.data_ptr<float>(),
      static_cast<int>(n2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void load_float4_sink(torch::Tensor x, torch::Tensor sink, int blocks, int threads) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK((x.numel() & 3) == 0, "n must be divisible by 4");
  const int64_t n4 = x.numel() >> 2;
  TORCH_CHECK(n4 <= static_cast<int64_t>(2147483647), "n/4 must fit int32");
  load_float4_sink_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const float4*>(x.data_ptr<float>()),
      sink.data_ptr<float>(),
      static_cast<int>(n4));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void load_float4_sink_var(torch::Tensor x, torch::Tensor sink, int blocks, int threads) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK((x.numel() & 3) == 0, "n must be divisible by 4");
  const int64_t n4 = x.numel() >> 2;
  TORCH_CHECK(n4 <= static_cast<int64_t>(2147483647), "n/4 must fit int32");
  load_float4_sink_var_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const float4*>(x.data_ptr<float>()),
      sink.data_ptr<float>(),
      static_cast<int>(n4));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void load_float4_asm_sink(torch::Tensor x, int blocks, int threads) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK((x.numel() & 3) == 0, "n must be divisible by 4");
  const int64_t n4 = x.numel() >> 2;
  TORCH_CHECK(n4 <= static_cast<int64_t>(2147483647), "n/4 must fit int32");
  load_float4_asm_sink_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const float4*>(x.data_ptr<float>()),
      static_cast<int>(n4));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void load_float4_block_chunk_sink(torch::Tensor x, torch::Tensor sink, int blocks, int threads) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK((x.numel() & 3) == 0, "n must be divisible by 4");
  const int64_t n4 = x.numel() >> 2;
  TORCH_CHECK(n4 <= static_cast<int64_t>(2147483647), "n/4 must fit int32");
  load_float4_block_chunk_sink_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const float4*>(x.data_ptr<float>()),
      sink.data_ptr<float>(),
      static_cast<int>(n4));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void load_float4_ldg_sink(torch::Tensor x, torch::Tensor sink, int blocks, int threads) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK((x.numel() & 3) == 0, "n must be divisible by 4");
  const int64_t n4 = x.numel() >> 2;
  TORCH_CHECK(n4 <= static_cast<int64_t>(2147483647), "n/4 must fit int32");
  load_float4_ldg_sink_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const float4*>(x.data_ptr<float>()),
      sink.data_ptr<float>(),
      static_cast<int>(n4));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void load_float4_cg_sink(torch::Tensor x, torch::Tensor sink, int blocks, int threads) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK((x.numel() & 3) == 0, "n must be divisible by 4");
  const int64_t n4 = x.numel() >> 2;
  TORCH_CHECK(n4 <= static_cast<int64_t>(2147483647), "n/4 must fit int32");
  load_float4_cg_sink_kernel<<<blocks, threads, 0, stream>>>(
      x.data_ptr<float>(), sink.data_ptr<float>(), static_cast<int>(n4));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void load_float4_ca_sink(torch::Tensor x, torch::Tensor sink, int blocks, int threads) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK((x.numel() & 3) == 0, "n must be divisible by 4");
  const int64_t n4 = x.numel() >> 2;
  TORCH_CHECK(n4 <= static_cast<int64_t>(2147483647), "n/4 must fit int32");
  load_float4_ca_sink_kernel<<<blocks, threads, 0, stream>>>(
      x.data_ptr<float>(), sink.data_ptr<float>(), static_cast<int>(n4));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void load_float4_tile4_sink(torch::Tensor x, torch::Tensor sink, int blocks, int threads) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK((x.numel() & 3) == 0, "n must be divisible by 4");
  const int64_t n4 = x.numel() >> 2;
  TORCH_CHECK(n4 <= static_cast<int64_t>(2147483647), "n/4 must fit int32");
  load_float4_tile4_sink_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const float4*>(x.data_ptr<float>()),
      sink.data_ptr<float>(),
      static_cast<int>(n4));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

    return load_inline(
        name="vectorsum_v2_a100_v14_bandwidth_sass_probe_ext",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-Xptxas=-v"],
        verbose=verbose_build,
    )


def timed(
    name: str,
    fn: Callable[[], None],
    sink: torch.Tensor,
    expected: torch.Tensor,
    bytes_moved: int,
    reps: int,
    warmups: int,
    checksum: bool,
) -> ProbeResult:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()

    durations = []
    for _ in range(reps):
        sink.zero_()
        clear_l2_cache()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        durations.append(start.elapsed_time(end) * 1000.0)

    checksum_value = None
    abs_err = None
    if checksum:
        checksum_tensor = sink.sum()
        checksum_value = float(checksum_tensor.item())
        abs_err = float(torch.abs(checksum_tensor - expected).item())

    return ProbeResult(
        name=name,
        mean_us=sum(durations) / len(durations),
        best_us=min(durations),
        worst_us=max(durations),
        bytes_moved=bytes_moved,
        checksum=checksum_value,
        abs_err=abs_err,
    )


def dump_sass(module_path: str, kernel_filters: list[str]) -> None:
    cuobjdump = os.getenv("CUOBJDUMP", "cuobjdump")
    try:
        result = subprocess.run(
            [cuobjdump, "--dump-sass", module_path],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"sass_dump_error={exc}", flush=True)
        return

    current: list[str] = []
    keep = False
    for line in result.stdout.splitlines():
        if "Function :" in line:
            if keep and current:
                print("\n".join(current), flush=True)
            current = [line]
            keep = any(pattern in line for pattern in kernel_filters)
        elif keep:
            current.append(line)
    if keep and current:
        print("\n".join(current), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="vectorsum_v2 A100 bandwidth/SASS probe")
    parser.add_argument("--size", type=int, default=52_428_800)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--blocks-per-sm", type=int, default=12)
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--dump-sass", action="store_true")
    parser.add_argument("--no-checksum", action="store_true")
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.cuda.init()
    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    blocks = sms * args.blocks_per_sm
    sink_count = blocks * args.threads
    input_bytes = args.size * 4
    sink_bytes = sink_count * 4

    print(
        f"gpu={props.name} sms={sms} size={args.size} seed={args.seed} "
        f"threads={args.threads} blocks_per_sm={args.blocks_per_sm} blocks={blocks} "
        f"sink_count={sink_count} reps={args.reps}",
        flush=True,
    )

    x = generate_input(args.size, args.seed)
    dst = torch.empty_like(x)
    sink = torch.empty(sink_count, device="cuda", dtype=torch.float32)
    expected = x.to(torch.float64).sum().to(torch.float32)
    ext = build_ext(args.verbose_build)
    module_path = getattr(ext, "__file__", "")
    print(f"extension={module_path}", flush=True)

    probes: list[tuple[str, Callable[[], None], int, bool]] = [
        ("empty_launch", lambda: ext.empty_launch(sink), 0, False),
        ("torch_copy_d2d", lambda: dst.copy_(x), input_bytes * 2, False),
        ("torch_sum_fp32", lambda: x.sum(), input_bytes, False),
        ("load_scalar_sink", lambda: ext.load_scalar_sink(x, sink, blocks, args.threads), input_bytes + sink_bytes, True),
        ("load_float2_sink", lambda: ext.load_float2_sink(x, sink, blocks, args.threads), input_bytes + sink_bytes, True),
        ("load_float4_sink", lambda: ext.load_float4_sink(x, sink, blocks, args.threads), input_bytes + sink_bytes, True),
        ("load_float4_sink_var", lambda: ext.load_float4_sink_var(x, sink, blocks, args.threads), input_bytes + sink_bytes, True),
        ("load_float4_asm_sink", lambda: ext.load_float4_asm_sink(x, blocks, args.threads), input_bytes, False),
        (
            "load_float4_block_chunk_sink",
            lambda: ext.load_float4_block_chunk_sink(x, sink, blocks, args.threads),
            input_bytes + sink_bytes,
            True,
        ),
        ("load_float4_ldg_sink", lambda: ext.load_float4_ldg_sink(x, sink, blocks, args.threads), input_bytes + sink_bytes, True),
        ("load_float4_cg_sink", lambda: ext.load_float4_cg_sink(x, sink, blocks, args.threads), input_bytes + sink_bytes, True),
        ("load_float4_ca_sink", lambda: ext.load_float4_ca_sink(x, sink, blocks, args.threads), input_bytes + sink_bytes, True),
        ("load_float4_tile4_sink", lambda: ext.load_float4_tile4_sink(x, sink, blocks, args.threads), input_bytes + sink_bytes, True),
    ]

    if args.only:
        selected = set(args.only)
        valid = {name for name, _, _, _ in probes}
        missing = selected - valid
        if missing:
            raise ValueError(f"unknown probe(s): {sorted(missing)}. valid probes: {sorted(valid)}")
        probes = [probe for probe in probes if probe[0] in selected]

    if args.dump_sass and module_path:
        dump_sass(
            module_path,
            [
                "load_scalar_sink_kernel",
                "load_float2_sink_kernel",
                "load_float4_sink_kernel",
                "load_float4_sink_var_kernel",
                "load_float4_asm_sink_kernel",
                "load_float4_block_chunk_sink_kernel",
                "load_float4_ldg_sink_kernel",
                "load_float4_cg_sink_kernel",
                "load_float4_ca_sink_kernel",
                "load_float4_tile4_sink_kernel",
            ],
        )

    for name, fn, bytes_moved, wants_checksum in probes:
        result = timed(
            name,
            fn,
            sink,
            expected,
            bytes_moved,
            args.reps,
            args.warmups,
            checksum=wants_checksum and not args.no_checksum,
        )
        check_text = ""
        if result.checksum is not None:
            ok = result.abs_err is not None and result.abs_err <= 1e-8 + 1e-5 * abs(float(expected.item()))
            check_text = f" checksum={result.checksum:.6g} abs_err={result.abs_err:.6g} ok={ok}"
        print(
            f"{result.name:24s} mean_us={result.mean_us:9.3f} "
            f"best_us={result.best_us:9.3f} worst_us={result.worst_us:9.3f} "
            f"bytes={result.bytes_moved:12d} gbps={result.gbps:8.1f}{check_text}",
            flush=True,
        )


if __name__ == "__main__":
    main()
