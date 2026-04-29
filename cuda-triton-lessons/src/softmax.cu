#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
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

enum class Version { V1_NAIVE, V2_FUSED, V3_ONLINE };

struct Config {
  int m = 4096;
  int n = 4096;
  int block_size = 256;
  int iterations = 50;
  int warmup = 10;
  bool csv = false;
  bool check = true;
  Version version = Version::V2_FUSED;
};

const char* version_name(Version v) {
  switch (v) {
    case Version::V1_NAIVE: return "v1_naive";
    case Version::V2_FUSED: return "v2_fused";
    case Version::V3_ONLINE: return "v3_online";
  }
  return "unknown";
}

Version parse_version(const std::string& s) {
  if (s == "1" || s == "v1" || s == "naive") return Version::V1_NAIVE;
  if (s == "2" || s == "v2" || s == "fused") return Version::V2_FUSED;
  if (s == "3" || s == "v3" || s == "online") return Version::V3_ONLINE;
  throw std::invalid_argument("unknown version: " + s);
}

}  // namespace

// -----------------------------------------------------------------------------
// Warp-level reduction primitives
// -----------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_max(float v) {
  for (int off = 16; off > 0; off >>= 1) {
    v = fmaxf(v, __shfl_down_sync(0xFFFFFFFFu, v, off));
  }
  return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down_sync(0xFFFFFFFFu, v, off);
  }
  return v;
}

// Online softmax merge within a warp: (m, s) is the running (max, sum_exp)
// pair. Merge rule: new_max = max(m1, m2),
//                  new_sum = s1 * exp(m1 - new_max) + s2 * exp(m2 - new_max).
// Guard on m == new_max avoids NaN when both are -inf (unused lanes).
__device__ __forceinline__ void warp_reduce_online(float& m, float& s) {
  for (int off = 16; off > 0; off >>= 1) {
    float other_m = __shfl_down_sync(0xFFFFFFFFu, m, off);
    float other_s = __shfl_down_sync(0xFFFFFFFFu, s, off);
    float new_max = fmaxf(m, other_m);
    float scale_self = (m == new_max) ? 1.0f : __expf(m - new_max);
    float scale_other = (other_m == new_max) ? 1.0f : __expf(other_m - new_max);
    s = s * scale_self + other_s * scale_other;
    m = new_max;
  }
}

// -----------------------------------------------------------------------------
// v1 — 3 separate kernels (naive, 4 HBM trips per element)
// -----------------------------------------------------------------------------
__global__ void softmax_v1_row_max(const float* __restrict__ in,
                                   float* __restrict__ row_max, int M, int N) {
  int row = blockIdx.x;
  if (row >= M) return;
  const float* x = in + row * N;
  int tid = threadIdx.x;
  int bs = blockDim.x;

  float local_max = -INFINITY;
  for (int i = tid; i < N; i += bs) {
    local_max = fmaxf(local_max, x[i]);
  }

  local_max = warp_reduce_max(local_max);

  __shared__ float scratch[32];
  int lane = tid & 31;
  int wid = tid >> 5;
  if (lane == 0) scratch[wid] = local_max;
  __syncthreads();
  if (wid == 0) {
    int num_warps = bs >> 5;
    local_max = (tid < num_warps) ? scratch[lane] : -INFINITY;
    local_max = warp_reduce_max(local_max);
    if (tid == 0) row_max[row] = local_max;
  }
}

__global__ void softmax_v1_row_sum(const float* __restrict__ in,
                                   const float* __restrict__ row_max,
                                   float* __restrict__ row_sum, int M, int N) {
  int row = blockIdx.x;
  if (row >= M) return;
  const float* x = in + row * N;
  int tid = threadIdx.x;
  int bs = blockDim.x;
  float m = row_max[row];

  float local_sum = 0.0f;
  for (int i = tid; i < N; i += bs) {
    local_sum += __expf(x[i] - m);
  }

  local_sum = warp_reduce_sum(local_sum);

  __shared__ float scratch[32];
  int lane = tid & 31;
  int wid = tid >> 5;
  if (lane == 0) scratch[wid] = local_sum;
  __syncthreads();
  if (wid == 0) {
    int num_warps = bs >> 5;
    local_sum = (tid < num_warps) ? scratch[lane] : 0.0f;
    local_sum = warp_reduce_sum(local_sum);
    if (tid == 0) row_sum[row] = local_sum;
  }
}

__global__ void softmax_v1_normalize(const float* __restrict__ in,
                                     const float* __restrict__ row_max,
                                     const float* __restrict__ row_sum,
                                     float* __restrict__ out, int M, int N) {
  int row = blockIdx.x;
  if (row >= M) return;
  const float* x = in + row * N;
  float* y = out + row * N;
  float m = row_max[row];
  float inv_s = 1.0f / row_sum[row];

  int tid = threadIdx.x;
  int bs = blockDim.x;
  for (int i = tid; i < N; i += bs) {
    y[i] = __expf(x[i] - m) * inv_s;
  }
}

// -----------------------------------------------------------------------------
// v2 — fused single-kernel with shared memory caching (2 HBM trips)
// -----------------------------------------------------------------------------
__global__ void softmax_v2_fused(const float* __restrict__ in,
                                 float* __restrict__ out, int M, int N) {
  extern __shared__ float sdata[];
  int row = blockIdx.x;
  if (row >= M) return;
  const float* x = in + row * N;
  float* y = out + row * N;
  int tid = threadIdx.x;
  int bs = blockDim.x;

  // Pass 1: load row into shared memory, compute local max
  float local_max = -INFINITY;
  for (int i = tid; i < N; i += bs) {
    float v = x[i];
    sdata[i] = v;
    local_max = fmaxf(local_max, v);
  }

  // Block reduce max
  local_max = warp_reduce_max(local_max);
  __shared__ float scratch[32];
  __shared__ float shared_max;
  int lane = tid & 31;
  int wid = tid >> 5;
  if (lane == 0) scratch[wid] = local_max;
  __syncthreads();
  if (wid == 0) {
    int num_warps = bs >> 5;
    local_max = (tid < num_warps) ? scratch[lane] : -INFINITY;
    local_max = warp_reduce_max(local_max);
    if (tid == 0) shared_max = local_max;
  }
  __syncthreads();
  float m = shared_max;

  // Pass 2: compute exp(x - m), overwrite sdata, accumulate sum
  float local_sum = 0.0f;
  for (int i = tid; i < N; i += bs) {
    float e = __expf(sdata[i] - m);
    sdata[i] = e;
    local_sum += e;
  }

  // Block reduce sum
  local_sum = warp_reduce_sum(local_sum);
  __shared__ float shared_sum;
  if (lane == 0) scratch[wid] = local_sum;
  __syncthreads();
  if (wid == 0) {
    int num_warps = bs >> 5;
    local_sum = (tid < num_warps) ? scratch[lane] : 0.0f;
    local_sum = warp_reduce_sum(local_sum);
    if (tid == 0) shared_sum = local_sum;
  }
  __syncthreads();
  float inv_s = 1.0f / shared_sum;

  // Pass 3: divide and write
  for (int i = tid; i < N; i += bs) {
    y[i] = sdata[i] * inv_s;
  }
}

// -----------------------------------------------------------------------------
// v3 — online softmax (1 streaming pass + normalize pass, no row caching)
// -----------------------------------------------------------------------------
__global__ void softmax_v3_online(const float* __restrict__ in,
                                  float* __restrict__ out, int M, int N) {
  int row = blockIdx.x;
  if (row >= M) return;
  const float* x = in + row * N;
  float* y = out + row * N;
  int tid = threadIdx.x;
  int bs = blockDim.x;

  // Pass 1: streaming online update of (max, sum)
  float local_max = -INFINITY;
  float local_sum = 0.0f;
  for (int i = tid; i < N; i += bs) {
    float v = x[i];
    float new_max = fmaxf(local_max, v);
    local_sum = local_sum * __expf(local_max - new_max) + __expf(v - new_max);
    // The __expf(local_max - new_max) is NaN-free because at iteration 1,
    // local_max transitions from -inf to v, but local_sum is still 0 so
    // 0 * NaN isn't encountered when the loop body is correctly ordered
    // with first update handled above. Safer to guard below — but the
    // streaming loop starts with local_max=-inf, local_sum=0; on first
    // iteration new_max = v, local_sum*exp(-inf-v) is 0*0 = 0 (not NaN since
    // -inf - v = -inf, not NaN, and exp(-inf)=0).
    local_max = new_max;
  }

  // Warp-level online merge
  warp_reduce_online(local_max, local_sum);

  __shared__ float max_scratch[32];
  __shared__ float sum_scratch[32];
  int lane = tid & 31;
  int wid = tid >> 5;
  if (lane == 0) {
    max_scratch[wid] = local_max;
    sum_scratch[wid] = local_sum;
  }
  __syncthreads();

  __shared__ float shared_max;
  __shared__ float shared_sum;
  if (wid == 0) {
    int num_warps = bs >> 5;
    if (tid < num_warps) {
      local_max = max_scratch[tid];
      local_sum = sum_scratch[tid];
    } else {
      local_max = -INFINITY;
      local_sum = 0.0f;
    }
    warp_reduce_online(local_max, local_sum);
    if (tid == 0) {
      shared_max = local_max;
      shared_sum = local_sum;
    }
  }
  __syncthreads();
  float m = shared_max;
  float inv_s = 1.0f / shared_sum;

  // Pass 2: re-read HBM, compute normalized output, write
  for (int i = tid; i < N; i += bs) {
    y[i] = __expf(x[i] - m) * inv_s;
  }
}

namespace {

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

    if (arg == "--m") {
      cfg.m = std::stoi(require_value(arg));
    } else if (arg == "--n") {
      cfg.n = std::stoi(require_value(arg));
    } else if (arg == "--block-size") {
      cfg.block_size = std::stoi(require_value(arg));
    } else if (arg == "--iterations") {
      cfg.iterations = std::stoi(require_value(arg));
    } else if (arg == "--warmup") {
      cfg.warmup = std::stoi(require_value(arg));
    } else if (arg == "--csv") {
      cfg.csv = true;
    } else if (arg == "--no-check") {
      cfg.check = false;
    } else if (arg == "--version") {
      cfg.version = parse_version(require_value(arg));
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: ./bin/softmax [--m M] [--n N] [--block-size BS] "
             "[--iterations I] [--warmup W] [--csv] [--no-check] "
             "[--version {1|2|3}]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }

  if (cfg.m <= 0 || cfg.n <= 0) {
    throw std::invalid_argument("--m, --n must be positive");
  }
  if (cfg.block_size <= 0 || cfg.block_size % 32 != 0 ||
      cfg.block_size > 1024) {
    throw std::invalid_argument(
        "--block-size must be multiple of 32 and <= 1024");
  }
  if (cfg.iterations <= 0 || cfg.warmup < 0) {
    throw std::invalid_argument("invalid timing configuration");
  }
  if (cfg.version == Version::V2_FUSED) {
    std::size_t smem_bytes = static_cast<std::size_t>(cfg.n) * sizeof(float);
    // T4 default per-block shared memory limit is 48 KB.
    if (smem_bytes > 48 * 1024) {
      throw std::invalid_argument(
          "v2 requires N <= 12288 so row fits in 48KB shared memory (got N=" +
          std::to_string(cfg.n) + ")");
    }
  }
  return cfg;
}

void cpu_softmax_reference(const std::vector<float>& in, std::vector<float>& out,
                           int M, int N) {
  for (int row = 0; row < M; ++row) {
    const float* x = in.data() + row * N;
    float* y = out.data() + row * N;

    double row_max = -std::numeric_limits<double>::infinity();
    for (int j = 0; j < N; ++j) {
      row_max = std::max(row_max, static_cast<double>(x[j]));
    }
    double row_sum = 0.0;
    for (int j = 0; j < N; ++j) {
      row_sum += std::exp(static_cast<double>(x[j]) - row_max);
    }
    for (int j = 0; j < N; ++j) {
      y[j] = static_cast<float>(std::exp(static_cast<double>(x[j]) - row_max) /
                                row_sum);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Config cfg = parse_args(argc, argv);

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    const std::size_t elems =
        static_cast<std::size_t>(cfg.m) * static_cast<std::size_t>(cfg.n);
    const std::size_t bytes = elems * sizeof(float);

    std::vector<float> h_in(elems);
    std::vector<float> h_out(elems);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto& x : h_in) x = dist(rng);

    float* d_in = nullptr;
    float* d_out = nullptr;
    float* d_row_max = nullptr;
    float* d_row_sum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_row_max, cfg.m * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_row_sum, cfg.m * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto launch = [&]() {
      dim3 grid(cfg.m);
      dim3 block(cfg.block_size);
      switch (cfg.version) {
        case Version::V1_NAIVE:
          softmax_v1_row_max<<<grid, block>>>(d_in, d_row_max, cfg.m, cfg.n);
          softmax_v1_row_sum<<<grid, block>>>(d_in, d_row_max, d_row_sum, cfg.m,
                                              cfg.n);
          softmax_v1_normalize<<<grid, block>>>(d_in, d_row_max, d_row_sum,
                                                d_out, cfg.m, cfg.n);
          break;
        case Version::V2_FUSED: {
          std::size_t smem =
              static_cast<std::size_t>(cfg.n) * sizeof(float);
          softmax_v2_fused<<<grid, block, smem>>>(d_in, d_out, cfg.m, cfg.n);
          break;
        }
        case Version::V3_ONLINE:
          softmax_v3_online<<<grid, block>>>(d_in, d_out, cfg.m, cfg.n);
          break;
      }
    };

    for (int i = 0; i < cfg.warmup; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> kernel_ms;
    kernel_ms.reserve(cfg.iterations);
    for (int i = 0; i < cfg.iterations; ++i) {
      CUDA_CHECK(cudaEventRecord(start));
      launch();
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
      kernel_ms.push_back(ms);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    bool checked = false;
    // CPU reference: at M*N <= 64M elements, CPU softmax takes a few seconds.
    if (cfg.check && elems <= (1ULL << 26)) {
      std::vector<float> h_ref(elems);
      cpu_softmax_reference(h_in, h_ref, cfg.m, cfg.n);
      for (std::size_t i = 0; i < elems; ++i) {
        float err = std::abs(h_out[i] - h_ref[i]);
        max_abs_err = std::max(max_abs_err, err);
        float ref_abs = std::abs(h_ref[i]);
        if (ref_abs > 1e-9f) {
          max_rel_err = std::max(max_rel_err, err / ref_abs);
        }
      }
      checked = true;
    }

    const auto min_it = std::min_element(kernel_ms.begin(), kernel_ms.end());
    double sum_ms = 0.0;
    for (float ms : kernel_ms) sum_ms += ms;
    const double avg_ms = sum_ms / static_cast<double>(kernel_ms.size());
    const double best_ms = static_cast<double>(*min_it);

    // HBM trips per element by version:
    //   v1: 3 reads + 1 write = 4 trips
    //   v2: 1 read  + 1 write = 2 trips
    //   v3: 2 reads + 1 write = 3 trips
    int hbm_trips = 0;
    switch (cfg.version) {
      case Version::V1_NAIVE: hbm_trips = 4; break;
      case Version::V2_FUSED: hbm_trips = 2; break;
      case Version::V3_ONLINE: hbm_trips = 3; break;
    }
    const double bytes_moved =
        static_cast<double>(elems) * sizeof(float) * hbm_trips;
    const double effective_gbps = bytes_moved / (best_ms / 1.0e3) / 1.0e9;
    const double theoretical_gbps =
        prop.memoryClockRate > 0
            ? static_cast<double>(prop.memoryClockRate) *
                  static_cast<double>(prop.memoryBusWidth) * 2.0 / (8.0 * 1.0e6)
            : 0.0;
    const double efficiency =
        theoretical_gbps > 0.0
            ? (effective_gbps / theoretical_gbps) * 100.0
            : 0.0;

    if (cfg.csv) {
      std::cout << "device,version,m,n,block_size,hbm_trips,best_ms,avg_ms,"
                   "effective_gbps,theoretical_gbps,efficiency_pct,"
                   "max_abs_error,max_rel_error,checked\n";
      std::cout << '"' << prop.name << '"' << ',' << version_name(cfg.version)
                << ',' << cfg.m << ',' << cfg.n << ',' << cfg.block_size << ','
                << hbm_trips << ',' << std::fixed << std::setprecision(6)
                << best_ms << ',' << avg_ms << ',' << effective_gbps << ','
                << theoretical_gbps << ',' << efficiency << ',' << max_abs_err
                << ',' << max_rel_err << ',' << (checked ? "yes" : "no")
                << '\n';
    } else {
      std::cout << "Softmax Benchmark\n";
      std::cout << "  device              : " << prop.name << '\n';
      std::cout << "  version             : " << version_name(cfg.version)
                << '\n';
      std::cout << "  M, N                : " << cfg.m << ", " << cfg.n << '\n';
      std::cout << "  block size          : " << cfg.block_size << '\n';
      std::cout << "  iterations          : " << cfg.iterations << '\n';
      std::cout << "  HBM trips per elem  : " << hbm_trips << '\n';
      std::cout << std::fixed << std::setprecision(3);
      std::cout << "  best kernel time ms : " << best_ms << '\n';
      std::cout << "  avg kernel time ms  : " << avg_ms << '\n';
      std::cout << "  effective GB/s      : " << effective_gbps << '\n';
      std::cout << "  theoretical GB/s    : " << theoretical_gbps << '\n';
      std::cout << "  efficiency %        : " << efficiency << '\n';
      std::cout << "  checked vs CPU      : " << (checked ? "yes" : "no") << '\n';
      if (checked) {
        std::cout << std::setprecision(8);
        std::cout << "  max abs error       : " << max_abs_err << '\n';
        std::cout << "  max rel error       : " << max_rel_err << '\n';
      }
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_row_max));
    CUDA_CHECK(cudaFree(d_row_sum));
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
