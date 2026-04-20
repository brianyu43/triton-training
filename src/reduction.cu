#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t status__ = (call);                                              \
    if (status__ != cudaSuccess) {                                              \
      throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                               std::to_string(__LINE__) + " -> " +              \
                               cudaGetErrorString(status__));                   \
    }                                                                           \
  } while (0)

namespace {

enum class Version { V1_ATOMIC, V2_SHARED, V3_UNROLL, V4_SHUFFLE, THRUST };

struct Config {
  std::size_t n = 1u << 24;
  int block_size = 256;
  int iterations = 50;
  int warmup = 10;
  bool csv = false;
  Version version = Version::V4_SHUFFLE;
};

const char* version_name(Version v) {
  switch (v) {
    case Version::V1_ATOMIC: return "v1_atomic";
    case Version::V2_SHARED: return "v2_shared";
    case Version::V3_UNROLL: return "v3_unroll";
    case Version::V4_SHUFFLE: return "v4_shuffle";
    case Version::THRUST: return "thrust";
  }
  return "unknown";
}

Version parse_version(const std::string& s) {
  if (s == "1" || s == "v1" || s == "atomic") return Version::V1_ATOMIC;
  if (s == "2" || s == "v2" || s == "shared") return Version::V2_SHARED;
  if (s == "3" || s == "v3" || s == "unroll") return Version::V3_UNROLL;
  if (s == "4" || s == "v4" || s == "shuffle") return Version::V4_SHUFFLE;
  if (s == "thrust") return Version::THRUST;
  throw std::invalid_argument("unknown version: " + s);
}

double theoretical_bandwidth_gbps(const cudaDeviceProp& prop) {
  if (prop.memoryClockRate == 0 || prop.memoryBusWidth == 0) {
    return 0.0;
  }
  return static_cast<double>(prop.memoryClockRate) *
         static_cast<double>(prop.memoryBusWidth) * 2.0 /
         (8.0 * 1.0e6);
}

__global__ void reduce_v1_atomic(const float* in, float* out, std::size_t n) {
  std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = idx; i < n; i += stride) {
    atomicAdd(out, in[i]);
  }
}

__global__ void reduce_v2_shared(const float* in, float* out, std::size_t n) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;

  float local = 0.0f;
  for (std::size_t i = idx; i < n; i += stride) {
    local += in[i];
  }
  sdata[tid] = local;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) atomicAdd(out, sdata[0]);
}

__global__ void reduce_v3_unroll(const float* in, float* out, std::size_t n) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;

  float local = 0.0f;
  for (std::size_t i = idx; i < n; i += stride) {
    local += in[i];
  }
  sdata[tid] = local;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid < 32) {
    float v = sdata[tid];
    if (blockDim.x >= 64) v += sdata[tid + 32];
    for (int offset = 16; offset > 0; offset >>= 1) {
      v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
    }
    if (tid == 0) atomicAdd(out, v);
  }
}

__global__ void reduce_v4_shuffle(const float* in, float* out, std::size_t n) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int lane = tid & 31;
  int wid = tid >> 5;
  std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;

  float local = 0.0f;
  for (std::size_t i = idx; i < n; i += stride) {
    local += in[i];
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    local += __shfl_down_sync(0xFFFFFFFFu, local, offset);
  }

  if (lane == 0) sdata[wid] = local;
  __syncthreads();

  int num_warps = blockDim.x >> 5;
  if (wid == 0) {
    float v = (tid < num_warps) ? sdata[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
    }
    if (tid == 0) atomicAdd(out, v);
  }
}

Config parse_args(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const std::string& flag) -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument("missing value for " + flag);
      }
      return argv[++i];
    };

    if (arg == "--n") {
      cfg.n = static_cast<std::size_t>(std::stoull(require_value(arg)));
    } else if (arg == "--block-size") {
      cfg.block_size = std::stoi(require_value(arg));
    } else if (arg == "--iterations") {
      cfg.iterations = std::stoi(require_value(arg));
    } else if (arg == "--warmup") {
      cfg.warmup = std::stoi(require_value(arg));
    } else if (arg == "--csv") {
      cfg.csv = true;
    } else if (arg == "--version") {
      cfg.version = parse_version(require_value(arg));
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: ./bin/reduction [--n N] [--block-size BS] "
             "[--iterations I] [--warmup W] [--csv] "
             "[--version {1|2|3|4|thrust}]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }

  if (cfg.n == 0) {
    throw std::invalid_argument("--n must be positive");
  }
  if (cfg.block_size <= 0 || cfg.block_size % 32 != 0) {
    throw std::invalid_argument("--block-size must be positive multiple of 32");
  }
  if (cfg.iterations <= 0 || cfg.warmup < 0) {
    throw std::invalid_argument("invalid timing configuration");
  }
  return cfg;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Config cfg = parse_args(argc, argv);

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    const std::size_t bytes = cfg.n * sizeof(float);

    std::vector<float> h_in(cfg.n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    double ref = 0.0;
    for (std::size_t i = 0; i < cfg.n; ++i) {
      h_in[i] = dist(rng);
      ref += static_cast<double>(h_in[i]);
    }

    float* d_in = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    const int max_blocks = prop.multiProcessorCount * 32;
    const int grid_size =
        std::min<int>((cfg.n + cfg.block_size - 1) / cfg.block_size, max_blocks);

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto launch = [&]() {
      switch (cfg.version) {
        case Version::V1_ATOMIC:
          reduce_v1_atomic<<<grid_size, cfg.block_size>>>(d_in, d_out, cfg.n);
          break;
        case Version::V2_SHARED: {
          std::size_t smem = cfg.block_size * sizeof(float);
          reduce_v2_shared<<<grid_size, cfg.block_size, smem>>>(d_in, d_out,
                                                                cfg.n);
          break;
        }
        case Version::V3_UNROLL: {
          std::size_t smem = cfg.block_size * sizeof(float);
          reduce_v3_unroll<<<grid_size, cfg.block_size, smem>>>(d_in, d_out,
                                                                cfg.n);
          break;
        }
        case Version::V4_SHUFFLE: {
          int num_warps = cfg.block_size / 32;
          std::size_t smem = num_warps * sizeof(float);
          reduce_v4_shuffle<<<grid_size, cfg.block_size, smem>>>(d_in, d_out,
                                                                 cfg.n);
          break;
        }
        case Version::THRUST:
          break;
      }
    };

    std::vector<float> kernel_ms;
    kernel_ms.reserve(cfg.iterations);
    float result = 0.0f;

    if (cfg.version == Version::THRUST) {
      thrust::device_ptr<float> d_ptr(d_in);
      for (int i = 0; i < cfg.warmup; ++i) {
        volatile float r = thrust::reduce(d_ptr, d_ptr + cfg.n, 0.0f);
        (void)r;
      }
      CUDA_CHECK(cudaDeviceSynchronize());

      for (int i = 0; i < cfg.iterations; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        float r = thrust::reduce(d_ptr, d_ptr + cfg.n, 0.0f);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        kernel_ms.push_back(ms);
        result = r;
      }
    } else {
      for (int i = 0; i < cfg.warmup; ++i) {
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        launch();
      }
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      for (int i = 0; i < cfg.iterations; ++i) {
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        kernel_ms.push_back(ms);
      }
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(
          cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    }

    const double abs_err = std::abs(static_cast<double>(result) - ref);

    const auto min_it = std::min_element(kernel_ms.begin(), kernel_ms.end());
    double sum_ms = 0.0;
    for (float ms : kernel_ms) sum_ms += ms;
    const double avg_ms = sum_ms / static_cast<double>(kernel_ms.size());
    const double best_ms = static_cast<double>(*min_it);

    const double bytes_read = static_cast<double>(cfg.n) * sizeof(float);
    const double effective_bandwidth =
        bytes_read / (best_ms / 1.0e3) / 1.0e9;
    const double theoretical_bandwidth = theoretical_bandwidth_gbps(prop);
    const double efficiency =
        theoretical_bandwidth > 0.0
            ? (effective_bandwidth / theoretical_bandwidth) * 100.0
            : 0.0;

    if (cfg.csv) {
      std::cout << "device,version,n,block_size,grid_size,best_ms,avg_ms,"
                   "effective_gbps,theoretical_gbps,efficiency_pct,"
                   "abs_error,result,reference\n";
      std::cout << '"' << prop.name << '"' << ',' << version_name(cfg.version)
                << ',' << cfg.n << ',' << cfg.block_size << ',' << grid_size
                << ',' << std::fixed << std::setprecision(6) << best_ms << ','
                << avg_ms << ',' << effective_bandwidth << ','
                << theoretical_bandwidth << ',' << efficiency << ',' << abs_err
                << ',' << result << ',' << ref << '\n';
    } else {
      std::cout << "Reduction Benchmark\n";
      std::cout << "  device              : " << prop.name << '\n';
      std::cout << "  version             : " << version_name(cfg.version)
                << '\n';
      std::cout << "  n                   : " << cfg.n << '\n';
      std::cout << "  block size          : " << cfg.block_size << '\n';
      std::cout << "  grid size           : " << grid_size << '\n';
      std::cout << "  iterations          : " << cfg.iterations << '\n';
      std::cout << "  warmup              : " << cfg.warmup << '\n';
      std::cout << std::fixed << std::setprecision(3);
      std::cout << "  best kernel time ms : " << best_ms << '\n';
      std::cout << "  avg kernel time ms  : " << avg_ms << '\n';
      std::cout << "  effective GB/s      : " << effective_bandwidth << '\n';
      std::cout << "  theoretical GB/s    : " << theoretical_bandwidth << '\n';
      std::cout << "  efficiency %        : " << efficiency << '\n';
      std::cout << std::setprecision(6);
      std::cout << "  result              : " << result << '\n';
      std::cout << "  reference           : " << ref << '\n';
      std::cout << "  abs error           : " << abs_err << '\n';
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
