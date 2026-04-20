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

enum class Version { NAIVE, FLASH };

// Head dim is fixed at 64 for this lesson.
constexpr int HEAD_DIM = 64;
constexpr int FA_BR = 64;   // Q rows per Flash block
constexpr int FA_BC = 32;   // K/V rows per Flash block

struct Config {
  int n = 2048;              // sequence length
  int iterations = 50;
  int warmup = 10;
  bool csv = false;
  bool check = true;
  Version version = Version::FLASH;
};

const char* version_name(Version v) {
  switch (v) {
    case Version::NAIVE: return "naive";
    case Version::FLASH: return "flash";
  }
  return "unknown";
}

Version parse_version(const std::string& s) {
  if (s == "naive" || s == "n") return Version::NAIVE;
  if (s == "flash" || s == "fa" || s == "f") return Version::FLASH;
  throw std::invalid_argument("unknown version: " + s);
}

}  // namespace

// -----------------------------------------------------------------------------
// Naive attention: three kernels
//   (1) scores S = Q @ K^T * scale          (N, N)
//   (2) P = softmax(S) row-wise             (N, N)
//   (3) O = P @ V                           (N, d)
// Materializes the N×N score matrix in HBM — the exact thing FA avoids.
// -----------------------------------------------------------------------------

// (1) Tiled matmul for S = Q @ K^T * scale.  Uses 32×32 shared-memory tiles.
__global__ void attn_naive_qk_scale(const float* __restrict__ Q,
                                    const float* __restrict__ K,
                                    float* __restrict__ S,
                                    int N, float scale) {
  constexpr int TILE = 32;
  __shared__ float Qs[TILE][TILE];
  __shared__ float Ks[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;  // Q row (S row)
  int col = blockIdx.x * TILE + threadIdx.x;  // S column (= K row for this output)

  float acc = 0.0f;
  // Tile walk across HEAD_DIM.
  //   Qs[i][j] = Q[by*TILE + i][t + j]         — thread (ty, tx) loads index (ty, tx)
  //   Ks[i][j] = K[bx*TILE + i][t + j]         — row index of K tile comes from ty,
  //                                              NOT tx (tx is the head-dim position).
  for (int t = 0; t < HEAD_DIM; t += TILE) {
    int qrow = blockIdx.y * TILE + threadIdx.y;
    int qcol = t + threadIdx.x;
    int krow = blockIdx.x * TILE + threadIdx.y;
    int kcol = t + threadIdx.x;
    Qs[threadIdx.y][threadIdx.x] =
        (qrow < N && qcol < HEAD_DIM) ? Q[qrow * HEAD_DIM + qcol] : 0.0f;
    Ks[threadIdx.y][threadIdx.x] =
        (krow < N && kcol < HEAD_DIM) ? K[krow * HEAD_DIM + kcol] : 0.0f;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE; ++k) {
      // Output thread (ty, tx) computes S[row][col] = sum_d Q[row][d] * K[col][d].
      // Q[row][t+k] = Qs[ty][k]; K[col][t+k] = Ks[tx][k] because Ks[i][*] is K row bx*TILE+i.
      acc += Qs[threadIdx.y][k] * Ks[threadIdx.x][k];
    }
    __syncthreads();
  }

  if (row < N && col < N) {
    S[row * N + col] = acc * scale;
  }
}

// (2) Row-wise softmax over S.  One block per row.  Shared-memory fused (like
// softmax v2) so we need the whole row to fit in 48 KB — caps N at 12288.
__global__ void attn_naive_softmax(const float* __restrict__ S,
                                   float* __restrict__ P, int N) {
  extern __shared__ float row_cache[];
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int bs = blockDim.x;

  const float* s_row = S + row * N;
  float* p_row = P + row * N;

  // Load row and find local max.
  float local_max = -INFINITY;
  for (int j = tid; j < N; j += bs) {
    float v = s_row[j];
    row_cache[j] = v;
    local_max = fmaxf(local_max, v);
  }
  __syncthreads();

  // Block reduce max.
  __shared__ float warp_max[32];
  __shared__ float warp_sum[32];
  unsigned mask = 0xFFFFFFFFu;
  float v = local_max;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_down_sync(mask, v, offset));
  }
  if ((tid & 31) == 0) warp_max[tid >> 5] = v;
  __syncthreads();

  if (tid < 32) {
    float w = (tid < (bs + 31) / 32) ? warp_max[tid] : -INFINITY;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      w = fmaxf(w, __shfl_down_sync(mask, w, offset));
    }
    if (tid == 0) warp_max[0] = w;
  }
  __syncthreads();
  float row_max = warp_max[0];

  // exp and local sum.
  float local_sum = 0.0f;
  for (int j = tid; j < N; j += bs) {
    float e = __expf(row_cache[j] - row_max);
    row_cache[j] = e;
    local_sum += e;
  }

  float s = local_sum;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    s += __shfl_down_sync(mask, s, offset);
  }
  if ((tid & 31) == 0) warp_sum[tid >> 5] = s;
  __syncthreads();

  if (tid < 32) {
    float w = (tid < (bs + 31) / 32) ? warp_sum[tid] : 0.0f;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      w += __shfl_down_sync(mask, w, offset);
    }
    if (tid == 0) warp_sum[0] = w;
  }
  __syncthreads();
  float row_sum = warp_sum[0];
  float inv_sum = 1.0f / row_sum;

  for (int j = tid; j < N; j += bs) {
    p_row[j] = row_cache[j] * inv_sum;
  }
}

// (3) Tiled matmul for O = P @ V.  Same 32×32 tile pattern, but now the
// K-dimension is N (sequence length), not HEAD_DIM.
__global__ void attn_naive_pv(const float* __restrict__ P,
                              const float* __restrict__ V,
                              float* __restrict__ O, int N) {
  constexpr int TILE = 32;
  __shared__ float Ps[TILE][TILE];
  __shared__ float Vs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;       // P row (= O row)
  int col = blockIdx.x * TILE + threadIdx.x;       // O col (< HEAD_DIM)

  float acc = 0.0f;
  for (int t = 0; t < N; t += TILE) {
    int pcol = t + threadIdx.x;
    int vrow = t + threadIdx.y;
    Ps[threadIdx.y][threadIdx.x] =
        (row < N && pcol < N) ? P[row * N + pcol] : 0.0f;
    Vs[threadIdx.y][threadIdx.x] =
        (vrow < N && col < HEAD_DIM) ? V[vrow * HEAD_DIM + col] : 0.0f;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE; ++k) {
      acc += Ps[threadIdx.y][k] * Vs[k][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < N && col < HEAD_DIM) {
    O[row * HEAD_DIM + col] = acc;
  }
}

// -----------------------------------------------------------------------------
// Flash Attention v1 (simplified, single batch/head)
//
// Grid : (ceil(N / Br), 1, 1) — one block per Q row-block.
// Block: FA_BR threads — each thread owns one Q row of the tile.
//
// Shared memory: Qs (Br×d), Ks (Bc×d), Vs (Bc×d).
// Per-thread registers: S_row[Bc], O_row[d], running (m, l).
//
// For each K/V block we compute a Br×Bc score tile, update running
// (max, sum) with the online-softmax formula (Lesson 05 v3) and accumulate
// into O with the rescale factor α = exp(m_old - m_new). Scores are NEVER
// written to HBM: they live and die inside this kernel. That is the whole
// point of Flash Attention — we trade O(N²) HBM traffic for recompute.
// -----------------------------------------------------------------------------
__global__ void flash_attention_v1(const float* __restrict__ Q,
                                   const float* __restrict__ K,
                                   const float* __restrict__ V,
                                   float* __restrict__ O,
                                   int N, float scale) {
  constexpr int Br = FA_BR;
  constexpr int Bc = FA_BC;
  constexpr int D  = HEAD_DIM;

  __shared__ float Qs[Br][D];
  __shared__ float Ks[Bc][D];
  __shared__ float Vs[Bc][D];

  int q_block = blockIdx.x;
  int tid     = threadIdx.x;              // 0..Br-1; owns Q row (q_block*Br + tid)
  int q_row   = q_block * Br + tid;
  bool active = (q_row < N);

  // Load Q block: Br * D elements, Br threads → D elements per thread.
#pragma unroll
  for (int d = 0; d < D; ++d) {
    Qs[tid][d] = active ? Q[q_row * D + d] : 0.0f;
  }
  __syncthreads();

  // Running state (registers, per thread = per Q row).
  float m_i = -INFINITY;
  float l_i = 0.0f;
  float O_row[D];
#pragma unroll
  for (int d = 0; d < D; ++d) O_row[d] = 0.0f;

  int n_kv_blocks = (N + Bc - 1) / Bc;
  for (int kv = 0; kv < n_kv_blocks; ++kv) {
    // Collaboratively load K, V blocks.  Br threads, Bc*D = 32*64 = 2048 elems
    // each for K and V → 2048/Br = 32 elements per thread per buffer.
    for (int i = tid; i < Bc * D; i += Br) {
      int r  = i / D;
      int c  = i % D;
      int kr = kv * Bc + r;
      Ks[r][c] = (kr < N) ? K[kr * D + c] : 0.0f;
      Vs[r][c] = (kr < N) ? V[kr * D + c] : 0.0f;
    }
    __syncthreads();

    // S row = Q[tid] · K[j]^T * scale, for j = 0..Bc-1. Kept in registers.
    float S_row[Bc];
    float m_tile = -INFINITY;
#pragma unroll
    for (int j = 0; j < Bc; ++j) {
      float s = 0.0f;
#pragma unroll
      for (int d = 0; d < D; ++d) {
        s += Qs[tid][d] * Ks[j][d];
      }
      s *= scale;
      // Mask out tail rows beyond N so they don't contribute to max/sum.
      int kr = kv * Bc + j;
      if (kr >= N) s = -INFINITY;
      S_row[j]  = s;
      m_tile    = fmaxf(m_tile, s);
    }

    // m_new = max(m_i, m_tile).  Guard against -inf arithmetic when this
    // thread's row has no valid columns yet (m_i == -inf on first iter).
    float m_new = fmaxf(m_i, m_tile);

    float alpha     = (m_i == -INFINITY) ? 0.0f : __expf(m_i - m_new);
    float l_tile    = 0.0f;

    // P_row = exp(S_row - m_new); accumulate rowsum. Reuse S_row storage.
#pragma unroll
    for (int j = 0; j < Bc; ++j) {
      float p = (S_row[j] == -INFINITY) ? 0.0f : __expf(S_row[j] - m_new);
      S_row[j] = p;
      l_tile  += p;
    }
    float l_new = alpha * l_i + l_tile;

    // O_new = alpha * O + P @ V.  For each output column, dot P_row into Vs[:,d].
#pragma unroll
    for (int d = 0; d < D; ++d) {
      float pv = 0.0f;
#pragma unroll
      for (int j = 0; j < Bc; ++j) {
        pv += S_row[j] * Vs[j][d];
      }
      O_row[d] = alpha * O_row[d] + pv;
    }

    m_i = m_new;
    l_i = l_new;
    __syncthreads();
  }

  // Final normalize and write.
  if (active) {
    float inv_l = 1.0f / l_i;
#pragma unroll
    for (int d = 0; d < D; ++d) {
      O[q_row * D + d] = O_row[d] * inv_l;
    }
  }
}

// -----------------------------------------------------------------------------
// Host helpers
// -----------------------------------------------------------------------------
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

    if (arg == "--n") {
      cfg.n = std::stoi(require_value(arg));
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
          << "Usage: ./bin/flash_attention [--n N] [--iterations I] "
             "[--warmup W] [--csv] [--no-check] [--version {naive|flash}]\n"
             "  Head dim is fixed at 64. Single batch/head. FP32.\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }

  if (cfg.n <= 0) throw std::invalid_argument("--n must be positive");
  if (cfg.iterations <= 0 || cfg.warmup < 0) {
    throw std::invalid_argument("invalid timing configuration");
  }
  if (cfg.version == Version::NAIVE && cfg.n > 12288) {
    throw std::invalid_argument(
        "naive softmax requires N <= 12288 (48KB smem row limit); got N=" +
        std::to_string(cfg.n));
  }
  return cfg;
}

// Double-precision CPU reference for attention. Single batch/head.
void cpu_attention_reference(const std::vector<float>& Q,
                             const std::vector<float>& K,
                             const std::vector<float>& V,
                             std::vector<float>& O, int N, float scale) {
  std::vector<double> S(static_cast<std::size_t>(N) * N);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      double acc = 0.0;
      for (int d = 0; d < HEAD_DIM; ++d) {
        acc += static_cast<double>(Q[i * HEAD_DIM + d]) *
               static_cast<double>(K[j * HEAD_DIM + d]);
      }
      S[i * N + j] = acc * static_cast<double>(scale);
    }
  }
  for (int i = 0; i < N; ++i) {
    double row_max = -std::numeric_limits<double>::infinity();
    for (int j = 0; j < N; ++j) row_max = std::max(row_max, S[i * N + j]);
    double row_sum = 0.0;
    for (int j = 0; j < N; ++j) {
      double e = std::exp(S[i * N + j] - row_max);
      S[i * N + j] = e;
      row_sum += e;
    }
    double inv = 1.0 / row_sum;
    for (int d = 0; d < HEAD_DIM; ++d) {
      double acc = 0.0;
      for (int j = 0; j < N; ++j) {
        acc += S[i * N + j] * inv * static_cast<double>(V[j * HEAD_DIM + d]);
      }
      O[i * HEAD_DIM + d] = static_cast<float>(acc);
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

    const std::size_t qkv_elems =
        static_cast<std::size_t>(cfg.n) * HEAD_DIM;
    const std::size_t qkv_bytes = qkv_elems * sizeof(float);
    const std::size_t s_bytes =
        static_cast<std::size_t>(cfg.n) * cfg.n * sizeof(float);

    std::vector<float> h_Q(qkv_elems), h_K(qkv_elems), h_V(qkv_elems);
    std::vector<float> h_O(qkv_elems), h_ref(qkv_elems);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : h_Q) x = dist(rng);
    for (auto& x : h_K) x = dist(rng);
    for (auto& x : h_V) x = dist(rng);

    float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

    float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr;
    float *d_O = nullptr, *d_S = nullptr, *d_P = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Q, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_K, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_V, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_O, qkv_bytes));
    if (cfg.version == Version::NAIVE) {
      CUDA_CHECK(cudaMalloc(&d_S, s_bytes));
      CUDA_CHECK(cudaMalloc(&d_P, s_bytes));
    }

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), qkv_bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto launch = [&]() {
      if (cfg.version == Version::NAIVE) {
        dim3 qk_block(32, 32);
        dim3 qk_grid((cfg.n + 31) / 32, (cfg.n + 31) / 32);
        attn_naive_qk_scale<<<qk_grid, qk_block>>>(d_Q, d_K, d_S, cfg.n, scale);

        dim3 sm_block(256);
        dim3 sm_grid(cfg.n);
        std::size_t smem = static_cast<std::size_t>(cfg.n) * sizeof(float);
        attn_naive_softmax<<<sm_grid, sm_block, smem>>>(d_S, d_P, cfg.n);

        dim3 pv_block(32, 32);
        dim3 pv_grid((HEAD_DIM + 31) / 32, (cfg.n + 31) / 32);
        attn_naive_pv<<<pv_grid, pv_block>>>(d_P, d_V, d_O, cfg.n);
      } else {
        dim3 block(FA_BR);
        dim3 grid((cfg.n + FA_BR - 1) / FA_BR);
        flash_attention_v1<<<grid, block>>>(d_Q, d_K, d_V, d_O, cfg.n, scale);
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
    CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, qkv_bytes, cudaMemcpyDeviceToHost));

    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    bool checked = false;
    // CPU reference is O(N² d) + O(N²) work. Cap at N=2048 (~0.5s).
    if (cfg.check && cfg.n <= 2048) {
      cpu_attention_reference(h_Q, h_K, h_V, h_ref, cfg.n, scale);
      for (std::size_t i = 0; i < qkv_elems; ++i) {
        float err = std::abs(h_O[i] - h_ref[i]);
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
    const double avg_ms  = sum_ms / static_cast<double>(kernel_ms.size());
    const double best_ms = static_cast<double>(*min_it);

    // HBM traffic accounting (per invocation, in bytes of FP32).
    //
    // Naive:
    //   (1) Q + K read  = 2 * N * d,     S written  = N²
    //   (2) S read + written again (softmax) ≈ 2 * N²,   P written = N²
    //       (coalesced into one ≈ 3 N² trip approximation: read S, write P,
    //        then read P for next kernel = 2 N² read + 2 N² write = 4 N²,
    //        but softmax itself is 1 read + 1 write = 2 N²)
    //   (3) P read = N²,    V read = N * d,    O written = N * d
    //   Total ≈ 4 N² + 4 N d (dominant term is 4 N²).
    //
    // Flash:
    //   Q, K, V read once each      = 3 * N * d
    //   O written once              =     N * d
    //   Total                       = 4 * N * d
    //
    // That ratio — 4 N² vs 4 N d — is the whole FA story.
    const double Nd = static_cast<double>(cfg.n) * HEAD_DIM;
    const double Nsq = static_cast<double>(cfg.n) * cfg.n;
    double bytes_moved;
    if (cfg.version == Version::NAIVE) {
      bytes_moved = (4.0 * Nsq + 4.0 * Nd) * sizeof(float);
    } else {
      bytes_moved = (4.0 * Nd) * sizeof(float);
    }
    const double effective_gbps = bytes_moved / (best_ms / 1.0e3) / 1.0e9;
    const double theoretical_gbps =
        prop.memoryClockRate > 0
            ? static_cast<double>(prop.memoryClockRate) *
                  static_cast<double>(prop.memoryBusWidth) * 2.0 / (8.0 * 1.0e6)
            : 0.0;
    const double efficiency =
        theoretical_gbps > 0.0 ? (effective_gbps / theoretical_gbps) * 100.0
                               : 0.0;

    // FLOPs (attention = 2 matmuls + 1 softmax).
    //   Q@K^T : 2 * N * N * d
    //   P@V   : 2 * N * N * d
    //   softmax ~ 3 N² (rough)
    // Total ≈ 4 N² d + 3 N² ≈ 4 N² d for d >> 1.
    const double flops = 4.0 * Nsq * HEAD_DIM + 3.0 * Nsq;
    const double gflops = flops / (best_ms / 1.0e3) / 1.0e9;

    if (cfg.csv) {
      std::cout << "device,version,n,head_dim,bytes_moved,best_ms,avg_ms,"
                   "effective_gbps,theoretical_gbps,efficiency_pct,gflops,"
                   "max_abs_error,max_rel_error,checked\n";
      std::cout << '"' << prop.name << '"' << ',' << version_name(cfg.version)
                << ',' << cfg.n << ',' << HEAD_DIM << ',' << std::fixed
                << std::setprecision(0) << bytes_moved << ','
                << std::setprecision(6) << best_ms << ',' << avg_ms << ','
                << effective_gbps << ',' << theoretical_gbps << ','
                << efficiency << ',' << gflops << ',' << max_abs_err << ','
                << max_rel_err << ',' << (checked ? "yes" : "no") << '\n';
    } else {
      std::cout << "Flash Attention Benchmark\n";
      std::cout << "  device              : " << prop.name << '\n';
      std::cout << "  version             : " << version_name(cfg.version)
                << '\n';
      std::cout << "  N, head_dim         : " << cfg.n << ", " << HEAD_DIM
                << '\n';
      std::cout << "  iterations          : " << cfg.iterations << '\n';
      std::cout << std::fixed << std::setprecision(3);
      std::cout << "  best kernel time ms : " << best_ms << '\n';
      std::cout << "  avg kernel time ms  : " << avg_ms << '\n';
      std::cout << "  bytes moved (HBM)   : " << bytes_moved << '\n';
      std::cout << "  effective GB/s      : " << effective_gbps << '\n';
      std::cout << "  theoretical GB/s    : " << theoretical_gbps << '\n';
      std::cout << "  efficiency %        : " << efficiency << '\n';
      std::cout << "  GFLOP/s             : " << gflops << '\n';
      std::cout << "  checked vs CPU      : " << (checked ? "yes" : "no")
                << '\n';
      if (checked) {
        std::cout << std::setprecision(8);
        std::cout << "  max abs error       : " << max_abs_err << '\n';
        std::cout << "  max rel error       : " << max_rel_err << '\n';
      }
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    if (d_S) CUDA_CHECK(cudaFree(d_S));
    if (d_P) CUDA_CHECK(cudaFree(d_P));
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
