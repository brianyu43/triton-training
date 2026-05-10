#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

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

enum class Version { V1_NAIVE, V2_TILED, V3_REGISTER, V4_TENSOR };

constexpr int TILE = 32;

struct Config {
  int m = 1024;
  int n = 1024;
  int k = 1024;
  int iterations = 20;
  int warmup = 5;
  bool csv = false;
  bool check = true;
  Version version = Version::V2_TILED;
};

const char* version_name(Version v) {
  switch (v) {
    case Version::V1_NAIVE: return "v1_naive";
    case Version::V2_TILED: return "v2_tiled";
    case Version::V3_REGISTER: return "v3_register";
    case Version::V4_TENSOR: return "v4_tensor";
  }
  return "unknown";
}

Version parse_version(const std::string& s) {
  if (s == "1" || s == "v1" || s == "naive") return Version::V1_NAIVE;
  if (s == "2" || s == "v2" || s == "tiled") return Version::V2_TILED;
  if (s == "3" || s == "v3" || s == "register") return Version::V3_REGISTER;
  if (s == "4" || s == "v4" || s == "tensor" || s == "tc")
    return Version::V4_TENSOR;
  throw std::invalid_argument("unknown version: " + s);
}

__global__ void matmul_v1_naive(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

__global__ void matmul_v2_tiled(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C, int M, int N, int K) {
  __shared__ float sA[TILE][TILE];
  __shared__ float sB[TILE][TILE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = blockIdx.y * TILE + ty;
  int col = blockIdx.x * TILE + tx;

  float sum = 0.0f;

  for (int tk = 0; tk < K; tk += TILE) {
    int a_col = tk + tx;
    sA[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

    int b_row = tk + ty;
    sB[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < TILE; ++kk) {
      sum += sA[ty][kk] * sB[kk][tx];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

// v3 — register blocking. Each thread computes TM x TN output elements,
// held in registers. Block tile BM x BN, K tile BK.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_v3_register(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C, int M, int N, int K) {
  __shared__ float sA[BM][BK];
  __shared__ float sB[BK][BN];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = ty * blockDim.x + tx;
  int threads_per_block = blockDim.x * blockDim.y;

  int block_row = blockIdx.y * BM;
  int block_col = blockIdx.x * BN;

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      acc[i][j] = 0.0f;
    }
  }

  constexpr int loads_per_thread_A = (BM * BK + 255) / 256;
  constexpr int loads_per_thread_B = (BK * BN + 255) / 256;

  for (int tk = 0; tk < K; tk += BK) {
    // Cooperative load of A tile [BM x BK]
#pragma unroll
    for (int i = 0; i < loads_per_thread_A; ++i) {
      int idx = tid + i * threads_per_block;
      if (idx < BM * BK) {
        int r = idx / BK;
        int c = idx % BK;
        int a_row = block_row + r;
        int a_col = tk + c;
        sA[r][c] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
      }
    }

    // Cooperative load of B tile [BK x BN]
#pragma unroll
    for (int i = 0; i < loads_per_thread_B; ++i) {
      int idx = tid + i * threads_per_block;
      if (idx < BK * BN) {
        int r = idx / BN;
        int c = idx % BN;
        int b_row = tk + r;
        int b_col = block_col + c;
        sB[r][c] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
      }
    }

    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float a_reg[TM];
      float b_reg[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        a_reg[i] = sA[ty * TM + i][kk];
      }
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        b_reg[j] = sB[kk][tx * TN + j];
      }
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          acc[i][j] += a_reg[i] * b_reg[j];
        }
      }
    }

    __syncthreads();
  }

  // Write output
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int c_row = block_row + ty * TM + i;
    if (c_row < M) {
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        int c_col = block_col + tx * TN + j;
        if (c_col < N) {
          C[c_row * N + c_col] = acc[i][j];
        }
      }
    }
  }
}

// v4 — Tensor Core via WMMA API. FP16 inputs, FP32 accumulator.
// Block tile: 64 x 64. 4 warps/block, each warp a 32 x 32 sub-tile (= 2x2 mma).
__global__ void matmul_v4_tc(const __half* __restrict__ A,
                             const __half* __restrict__ B,
                             float* __restrict__ C, int M, int N, int K) {
  using namespace nvcuda::wmma;
  constexpr int BM = 64, BN = 64, BK = 16;
  constexpr int WM = 32, WN = 32;
  constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;

  __shared__ __half sA[BM][BK];
  __shared__ __half sB[BK][BN];

  int tid = threadIdx.x;
  int warp_id = tid / 32;
  int warp_row = warp_id / 2;
  int warp_col = warp_id % 2;

  int block_row = blockIdx.y * BM;
  int block_col = blockIdx.x * BN;
  int warp_row_start = block_row + warp_row * WM;
  int warp_col_start = block_col + warp_col * WN;

  fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      fill_fragment(acc[i][j], 0.0f);
    }
  }

  constexpr int THREADS = 128;
  constexpr int LOADS_A = BM * BK / THREADS;  // 64*16/128 = 8
  constexpr int LOADS_B = BK * BN / THREADS;  // 16*64/128 = 8

  for (int tk = 0; tk < K; tk += BK) {
#pragma unroll
    for (int i = 0; i < LOADS_A; ++i) {
      int idx = tid + i * THREADS;
      int r = idx / BK;
      int c = idx % BK;
      int a_row = block_row + r;
      int a_col = tk + c;
      sA[r][c] = (a_row < M && a_col < K) ? A[a_row * K + a_col]
                                          : __float2half(0.0f);
    }
#pragma unroll
    for (int i = 0; i < LOADS_B; ++i) {
      int idx = tid + i * THREADS;
      int r = idx / BN;
      int c = idx % BN;
      int b_row = tk + r;
      int b_col = block_col + c;
      sB[r][c] = (b_row < K && b_col < N) ? B[b_row * N + b_col]
                                          : __float2half(0.0f);
    }

    __syncthreads();

    // Each warp computes 2x2 WMMA tiles (32x32 subtile)
#pragma unroll
    for (int kk = 0; kk < BK; kk += WMMA_K) {
      fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag[2];
      fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> b_frag[2];

#pragma unroll
      for (int i = 0; i < 2; ++i) {
        int row_off = warp_row * WM + i * WMMA_M;
        load_matrix_sync(a_frag[i], &sA[row_off][kk], BK);
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        int col_off = warp_col * WN + j * WMMA_N;
        load_matrix_sync(b_frag[j], &sB[kk][col_off], BN);
      }

#pragma unroll
      for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
      }
    }

    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      int row = warp_row_start + i * WMMA_M;
      int col = warp_col_start + j * WMMA_N;
      if (row < M && col < N) {
        store_matrix_sync(&C[row * N + col], acc[i][j], N, mem_row_major);
      }
    }
  }
}

__global__ void float_to_half_kernel(const float* src, __half* dst,
                                     std::size_t n) {
  std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = __float2half(src[idx]);
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

    if (arg == "--m") {
      cfg.m = std::stoi(require_value(arg));
    } else if (arg == "--n") {
      cfg.n = std::stoi(require_value(arg));
    } else if (arg == "--k") {
      cfg.k = std::stoi(require_value(arg));
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
          << "Usage: ./bin/matmul [--m M] [--n N] [--k K] "
             "[--iterations I] [--warmup W] [--csv] [--no-check] "
             "[--version {1|2|3|4}]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }

  if (cfg.m <= 0 || cfg.n <= 0 || cfg.k <= 0) {
    throw std::invalid_argument("--m, --n, --k must be positive");
  }
  if (cfg.iterations <= 0 || cfg.warmup < 0) {
    throw std::invalid_argument("invalid timing configuration");
  }
  if (cfg.version == Version::V4_TENSOR) {
    if (cfg.m % 64 != 0 || cfg.n % 64 != 0 || cfg.k % 16 != 0) {
      throw std::invalid_argument(
          "v4 requires M, N multiples of 64 and K multiple of 16");
    }
  }
  return cfg;
}

void cpu_matmul_reference(const std::vector<float>& A,
                          const std::vector<float>& B, std::vector<float>& C,
                          int M, int N, int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double sum = 0.0;
      for (int kk = 0; kk < K; ++kk) {
        sum += static_cast<double>(A[i * K + kk]) *
               static_cast<double>(B[kk * N + j]);
      }
      C[i * N + j] = static_cast<float>(sum);
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

    const std::size_t elems_A = static_cast<std::size_t>(cfg.m) * cfg.k;
    const std::size_t elems_B = static_cast<std::size_t>(cfg.k) * cfg.n;
    const std::size_t elems_C = static_cast<std::size_t>(cfg.m) * cfg.n;
    const std::size_t bytes_A = elems_A * sizeof(float);
    const std::size_t bytes_B = elems_B * sizeof(float);
    const std::size_t bytes_C = elems_C * sizeof(float);

    std::vector<float> h_A(elems_A);
    std::vector<float> h_B(elems_B);
    std::vector<float> h_C(elems_C);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : h_A) x = dist(rng);
    for (auto& x : h_B) x = dist(rng);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));

    __half* d_A_half = nullptr;
    __half* d_B_half = nullptr;
    if (cfg.version == Version::V4_TENSOR) {
      CUDA_CHECK(cudaMalloc(&d_A_half, elems_A * sizeof(__half)));
      CUDA_CHECK(cudaMalloc(&d_B_half, elems_B * sizeof(__half)));
      int threads = 256;
      int blocks_A = static_cast<int>((elems_A + threads - 1) / threads);
      int blocks_B = static_cast<int>((elems_B + threads - 1) / threads);
      float_to_half_kernel<<<blocks_A, threads>>>(d_A, d_A_half, elems_A);
      float_to_half_kernel<<<blocks_B, threads>>>(d_B, d_B_half, elems_B);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto launch = [&]() {
      if (cfg.version == Version::V1_NAIVE) {
        dim3 block(16, 16);
        dim3 grid((cfg.n + block.x - 1) / block.x,
                  (cfg.m + block.y - 1) / block.y);
        matmul_v1_naive<<<grid, block>>>(d_A, d_B, d_C, cfg.m, cfg.n, cfg.k);
      } else if (cfg.version == Version::V2_TILED) {
        dim3 block(TILE, TILE);
        dim3 grid((cfg.n + TILE - 1) / TILE, (cfg.m + TILE - 1) / TILE);
        matmul_v2_tiled<<<grid, block>>>(d_A, d_B, d_C, cfg.m, cfg.n, cfg.k);
      } else if (cfg.version == Version::V3_REGISTER) {
        constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
        dim3 block(BN / TN, BM / TM);  // 16 x 16 = 256 threads
        dim3 grid((cfg.n + BN - 1) / BN, (cfg.m + BM - 1) / BM);
        matmul_v3_register<BM, BN, BK, TM, TN>
            <<<grid, block>>>(d_A, d_B, d_C, cfg.m, cfg.n, cfg.k);
      } else {
        dim3 block(128);  // 4 warps
        dim3 grid(cfg.n / 64, cfg.m / 64);
        matmul_v4_tc<<<grid, block>>>(d_A_half, d_B_half, d_C, cfg.m, cfg.n,
                                      cfg.k);
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
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    bool checked = false;
    const double total_mul = static_cast<double>(cfg.m) *
                             static_cast<double>(cfg.n) *
                             static_cast<double>(cfg.k);

    if (cfg.check && total_mul <= 1.1e9) {
      std::vector<float> h_ref(elems_C);
      cpu_matmul_reference(h_A, h_B, h_ref, cfg.m, cfg.n, cfg.k);
      for (std::size_t i = 0; i < h_C.size(); ++i) {
        float err = std::abs(h_C[i] - h_ref[i]);
        max_abs_err = std::max(max_abs_err, err);
        float ref_abs = std::abs(h_ref[i]);
        if (ref_abs > 1e-6f) {
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

    const double flops = 2.0 * total_mul;
    const double effective_tflops = (flops / (best_ms / 1.0e3)) / 1.0e12;

    if (cfg.csv) {
      std::cout << "device,version,m,n,k,best_ms,avg_ms,effective_tflops,"
                   "max_abs_error,max_rel_error,checked\n";
      std::cout << '"' << prop.name << '"' << ',' << version_name(cfg.version)
                << ',' << cfg.m << ',' << cfg.n << ',' << cfg.k << ','
                << std::fixed << std::setprecision(6) << best_ms << ','
                << avg_ms << ',' << effective_tflops << ',' << max_abs_err
                << ',' << max_rel_err << ',' << (checked ? "yes" : "no")
                << '\n';
    } else {
      std::cout << "Matmul Benchmark\n";
      std::cout << "  device              : " << prop.name << '\n';
      std::cout << "  version             : " << version_name(cfg.version)
                << '\n';
      std::cout << "  M, N, K             : " << cfg.m << ", " << cfg.n << ", "
                << cfg.k << '\n';
      std::cout << "  iterations          : " << cfg.iterations << '\n';
      std::cout << "  warmup              : " << cfg.warmup << '\n';
      std::cout << std::fixed << std::setprecision(3);
      std::cout << "  best kernel time ms : " << best_ms << '\n';
      std::cout << "  avg kernel time ms  : " << avg_ms << '\n';
      std::cout << "  effective TFLOPS    : " << effective_tflops << '\n';
      std::cout << "  checked vs CPU      : "
                << (checked ? "yes" : "no (too large)") << '\n';
      if (checked) {
        std::cout << std::setprecision(8);
        std::cout << "  max abs error       : " << max_abs_err << '\n';
        std::cout << "  max rel error       : " << max_rel_err << '\n';
      }
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    if (d_A_half) CUDA_CHECK(cudaFree(d_A_half));
    if (d_B_half) CUDA_CHECK(cudaFree(d_B_half));
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
