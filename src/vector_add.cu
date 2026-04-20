#include <cuda_runtime.h>

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

struct Config {
  std::size_t n = 1u << 24;
  int block_size = 256;
  int iterations = 100;
  int warmup = 20;
  bool csv = false;
  bool pinned = true;
};

double theoretical_bandwidth_gbps(const cudaDeviceProp& prop) {
  if (prop.memoryClockRate == 0 || prop.memoryBusWidth == 0) {
    return 0.0;
  }
  return static_cast<double>(prop.memoryClockRate) *
         static_cast<double>(prop.memoryBusWidth) * 2.0 /
         (8.0 * 1.0e6);
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
    } else if (arg == "--pageable") {
      cfg.pinned = false;
    } else if (arg == "--pinned") {
      cfg.pinned = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: ./bin/vector_add [--n N] [--block-size BS] "
             "[--iterations I] [--warmup W] [--csv] [--pageable|--pinned]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }

  if (cfg.n == 0) {
    throw std::invalid_argument("--n must be positive");
  }
  if (cfg.block_size <= 0 || cfg.iterations <= 0 || cfg.warmup < 0) {
    throw std::invalid_argument("invalid launch or timing configuration");
  }
  return cfg;
}

__global__ void vector_add_kernel(const float* a, const float* b, float* c,
                                  std::size_t n) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;

  for (std::size_t i = idx; i < n; i += stride) {
    c[i] = a[i] + b[i];
  }
}

float max_abs_error(const std::vector<float>& ref, const std::vector<float>& got) {
  float max_err = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_err = std::max(max_err, std::abs(ref[i] - got[i]));
  }
  return max_err;
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

    std::vector<float> h_ref(cfg.n);
    std::vector<float> h_out(cfg.n);

    float* h_a = nullptr;
    float* h_b = nullptr;
    float* h_c = nullptr;
    if (cfg.pinned) {
      CUDA_CHECK(cudaMallocHost(&h_a, bytes));
      CUDA_CHECK(cudaMallocHost(&h_b, bytes));
      CUDA_CHECK(cudaMallocHost(&h_c, bytes));
    } else {
      h_a = new float[cfg.n];
      h_b = new float[cfg.n];
      h_c = new float[cfg.n];
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (std::size_t i = 0; i < cfg.n; ++i) {
      h_a[i] = dist(rng);
      h_b[i] = dist(rng);
      h_ref[i] = h_a[i] + h_b[i];
    }

    const int max_blocks = prop.multiProcessorCount * 32;
    const int grid_size =
        std::min<int>((cfg.n + cfg.block_size - 1) / cfg.block_size, max_blocks);

    cudaEvent_t transfer_start{};
    cudaEvent_t transfer_stop{};
    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&transfer_start));
    CUDA_CHECK(cudaEventCreate(&transfer_stop));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(transfer_start));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(transfer_stop));
    CUDA_CHECK(cudaEventSynchronize(transfer_stop));

    float h2d_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, transfer_start, transfer_stop));

    for (int i = 0; i < cfg.warmup; ++i) {
      vector_add_kernel<<<grid_size, cfg.block_size>>>(d_a, d_b, d_c, cfg.n);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> kernel_ms;
    kernel_ms.reserve(cfg.iterations);

    for (int i = 0; i < cfg.iterations; ++i) {
      CUDA_CHECK(cudaEventRecord(start));
      vector_add_kernel<<<grid_size, cfg.block_size>>>(d_a, d_b, d_c, cfg.n);
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));

      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
      kernel_ms.push_back(ms);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(transfer_start));
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(transfer_stop));
    CUDA_CHECK(cudaEventSynchronize(transfer_stop));

    float d2h_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, transfer_start, transfer_stop));

    std::copy(h_c, h_c + cfg.n, h_out.begin());
    const float max_err = max_abs_error(h_ref, h_out);

    const auto min_it = std::min_element(kernel_ms.begin(), kernel_ms.end());
    double sum_ms = 0.0;
    for (float ms : kernel_ms) {
      sum_ms += ms;
    }
    const double avg_ms = sum_ms / static_cast<double>(kernel_ms.size());
    const double best_ms = static_cast<double>(*min_it);

    const double bytes_moved = static_cast<double>(cfg.n) * sizeof(float) * 3.0;
    const double effective_bandwidth =
        bytes_moved / (best_ms / 1.0e3) / 1.0e9;
    const double theoretical_bandwidth = theoretical_bandwidth_gbps(prop);
    const double efficiency =
        theoretical_bandwidth > 0.0
            ? (effective_bandwidth / theoretical_bandwidth) * 100.0
            : 0.0;

    const char* mode_str = cfg.pinned ? "pinned" : "pageable";

    if (cfg.csv) {
      std::cout << "device,mode,n,block_size,grid_size,best_ms,avg_ms,"
                   "effective_gbps,theoretical_gbps,efficiency_pct,h2d_ms,"
                   "d2h_ms,max_abs_error\n";
      std::cout << '"' << prop.name << '"' << ',' << mode_str << ',' << cfg.n
                << ',' << cfg.block_size << ',' << grid_size << ',' << std::fixed
                << std::setprecision(6) << best_ms << ',' << avg_ms << ','
                << effective_bandwidth << ',' << theoretical_bandwidth << ','
                << efficiency << ',' << h2d_ms << ',' << d2h_ms << ',' << max_err
                << '\n';
    } else {
      std::cout << "Vector Add Benchmark\n";
      std::cout << "  device              : " << prop.name << '\n';
      std::cout << "  mode                : " << mode_str << '\n';
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
      std::cout << "  H2D copy ms         : " << h2d_ms << '\n';
      std::cout << "  D2H copy ms         : " << d2h_ms << '\n';
      std::cout << std::setprecision(8);
      std::cout << "  max abs error       : " << max_err << '\n';
    }

    CUDA_CHECK(cudaEventDestroy(transfer_start));
    CUDA_CHECK(cudaEventDestroy(transfer_stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    if (cfg.pinned) {
      CUDA_CHECK(cudaFreeHost(h_a));
      CUDA_CHECK(cudaFreeHost(h_b));
      CUDA_CHECK(cudaFreeHost(h_c));
    } else {
      delete[] h_a;
      delete[] h_b;
      delete[] h_c;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
