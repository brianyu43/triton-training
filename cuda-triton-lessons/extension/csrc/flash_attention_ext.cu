// Flash Attention PyTorch extension.
//
// This file packages the Lesson 06 kernels (flash_attention_v1 and the naive
// 3-kernel baseline) into PyTorch custom ops registered via TORCH_LIBRARY.
// Once the extension is imported, they are callable from Python as:
//
//     torch.ops.mylib.flash_attention(q, k, v)
//     torch.ops.mylib.naive_attention(q, k, v)
//
// Shape contract: Q, K, V are contiguous FP32 tensors of shape (N, d) with
// d == 64. Returns O of the same shape. Single batch/head — the caller is
// expected to squeeze batch/head dims before calling (the test harness does).

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// -----------------------------------------------------------------------------
// Kernel parameters (must match Lesson 06)
// -----------------------------------------------------------------------------
constexpr int HEAD_DIM = 64;
constexpr int FA_BR    = 64;
constexpr int FA_BC    = 32;

// -----------------------------------------------------------------------------
// Naive baseline kernels (copied from src/flash_attention.cu)
// -----------------------------------------------------------------------------
__global__ void attn_naive_qk_scale(const float* __restrict__ Q,
                                    const float* __restrict__ K,
                                    float* __restrict__ S,
                                    int N, float scale) {
  constexpr int TILE = 32;
  __shared__ float Qs[TILE][TILE];
  __shared__ float Ks[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;

  float acc = 0.0f;
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
      acc += Qs[threadIdx.y][k] * Ks[threadIdx.x][k];
    }
    __syncthreads();
  }

  if (row < N && col < N) {
    S[row * N + col] = acc * scale;
  }
}

__global__ void attn_naive_softmax(const float* __restrict__ S,
                                   float* __restrict__ P, int N) {
  extern __shared__ float row_cache[];
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int bs  = blockDim.x;

  const float* s_row = S + row * N;
  float* p_row = P + row * N;

  float local_max = -INFINITY;
  for (int j = tid; j < N; j += bs) {
    float v = s_row[j];
    row_cache[j] = v;
    local_max = fmaxf(local_max, v);
  }
  __syncthreads();

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

__global__ void attn_naive_pv(const float* __restrict__ P,
                              const float* __restrict__ V,
                              float* __restrict__ O, int N) {
  constexpr int TILE = 32;
  __shared__ float Ps[TILE][TILE];
  __shared__ float Vs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;

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
// Flash Attention v1 kernel (copied from src/flash_attention.cu)
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
  int tid     = threadIdx.x;
  int q_row   = q_block * Br + tid;
  bool active = (q_row < N);

#pragma unroll
  for (int d = 0; d < D; ++d) {
    Qs[tid][d] = active ? Q[q_row * D + d] : 0.0f;
  }
  __syncthreads();

  float m_i = -INFINITY;
  float l_i = 0.0f;
  float O_row[D];
#pragma unroll
  for (int d = 0; d < D; ++d) O_row[d] = 0.0f;

  int n_kv_blocks = (N + Bc - 1) / Bc;
  for (int kv = 0; kv < n_kv_blocks; ++kv) {
    for (int i = tid; i < Bc * D; i += Br) {
      int r  = i / D;
      int c  = i % D;
      int kr = kv * Bc + r;
      Ks[r][c] = (kr < N) ? K[kr * D + c] : 0.0f;
      Vs[r][c] = (kr < N) ? V[kr * D + c] : 0.0f;
    }
    __syncthreads();

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
      int kr = kv * Bc + j;
      if (kr >= N) s = -INFINITY;
      S_row[j]  = s;
      m_tile    = fmaxf(m_tile, s);
    }

    float m_new = fmaxf(m_i, m_tile);
    float alpha = (m_i == -INFINITY) ? 0.0f : __expf(m_i - m_new);
    float l_tile = 0.0f;

#pragma unroll
    for (int j = 0; j < Bc; ++j) {
      float p = (S_row[j] == -INFINITY) ? 0.0f : __expf(S_row[j] - m_new);
      S_row[j] = p;
      l_tile  += p;
    }
    float l_new = alpha * l_i + l_tile;

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

  if (active) {
    float inv_l = 1.0f / l_i;
#pragma unroll
    for (int d = 0; d < D; ++d) {
      O[q_row * D + d] = O_row[d] * inv_l;
    }
  }
}

// -----------------------------------------------------------------------------
// Host wrappers.  These are the C++ functions that PyTorch calls.  They:
//   (1) validate the tensors (shape, device, dtype, contiguity)
//   (2) allocate the output tensor via PyTorch (so it's a real torch.Tensor)
//   (3) pull raw pointers, grab the current CUDA stream
//   (4) launch our kernel on that stream
//   (5) return the output
//
// Everything Python-facing about PyTorch custom ops is in these ~30 lines.
// -----------------------------------------------------------------------------

static void check_qkv(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(),           name, " must be CUDA");
  TORCH_CHECK(t.is_contiguous(),     name, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == at::kFloat,
                                     name, " must be float32");
  TORCH_CHECK(t.dim() == 2,          name, " must be 2-D (N, d)");
  TORCH_CHECK(t.size(1) == HEAD_DIM, name, " must have d == ", HEAD_DIM);
}

torch::Tensor flash_attention_forward(const torch::Tensor& Q,
                                      const torch::Tensor& K,
                                      const torch::Tensor& V) {
  check_qkv(Q, "Q");
  check_qkv(K, "K");
  check_qkv(V, "V");
  const int N = static_cast<int>(Q.size(0));
  TORCH_CHECK(K.size(0) == N, "K N mismatch");
  TORCH_CHECK(V.size(0) == N, "V N mismatch");

  auto O = torch::empty({N, HEAD_DIM}, Q.options());

  float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
  dim3 block(FA_BR);
  dim3 grid((N + FA_BR - 1) / FA_BR);

  auto stream = at::cuda::getCurrentCUDAStream();
  flash_attention_v1<<<grid, block, 0, stream>>>(
      Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
      O.data_ptr<float>(), N, scale);

  AT_CUDA_CHECK(cudaGetLastError());
  return O;
}

torch::Tensor naive_attention_forward(const torch::Tensor& Q,
                                      const torch::Tensor& K,
                                      const torch::Tensor& V) {
  check_qkv(Q, "Q");
  check_qkv(K, "K");
  check_qkv(V, "V");
  const int N = static_cast<int>(Q.size(0));
  TORCH_CHECK(K.size(0) == N, "K N mismatch");
  TORCH_CHECK(V.size(0) == N, "V N mismatch");
  TORCH_CHECK(N <= 12288, "naive softmax requires N <= 12288 (48KB smem)");

  auto S = torch::empty({N, N},        Q.options());
  auto P = torch::empty({N, N},        Q.options());
  auto O = torch::empty({N, HEAD_DIM}, Q.options());

  float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
  auto stream = at::cuda::getCurrentCUDAStream();

  dim3 qk_block(32, 32);
  dim3 qk_grid((N + 31) / 32, (N + 31) / 32);
  attn_naive_qk_scale<<<qk_grid, qk_block, 0, stream>>>(
      Q.data_ptr<float>(), K.data_ptr<float>(), S.data_ptr<float>(), N, scale);

  dim3 sm_block(256);
  dim3 sm_grid(N);
  std::size_t smem = static_cast<std::size_t>(N) * sizeof(float);
  attn_naive_softmax<<<sm_grid, sm_block, smem, stream>>>(
      S.data_ptr<float>(), P.data_ptr<float>(), N);

  dim3 pv_block(32, 32);
  dim3 pv_grid((HEAD_DIM + 31) / 32, (N + 31) / 32);
  attn_naive_pv<<<pv_grid, pv_block, 0, stream>>>(
      P.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(), N);

  AT_CUDA_CHECK(cudaGetLastError());
  return O;
}

// -----------------------------------------------------------------------------
// torch.library registration.  After import, the ops are callable as:
//     torch.ops.mylib.flash_attention(q, k, v)
//     torch.ops.mylib.naive_attention(q, k, v)
// -----------------------------------------------------------------------------
TORCH_LIBRARY(mylib, m) {
  m.def("flash_attention(Tensor q, Tensor k, Tensor v) -> Tensor");
  m.def("naive_attention(Tensor q, Tensor k, Tensor v) -> Tensor");
}

TORCH_LIBRARY_IMPL(mylib, CUDA, m) {
  m.impl("flash_attention", &flash_attention_forward);
  m.impl("naive_attention", &naive_attention_forward);
}

// An empty pybind11 module so `import mylib_ext` works from Python; loading
// the .so alone is enough to trigger the TORCH_LIBRARY registrations above.
// Without this Python's import machinery errors with "no PyInit_mylib_ext".
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
