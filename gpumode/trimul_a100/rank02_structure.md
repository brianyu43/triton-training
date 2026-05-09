# Rank #2 CUDA Extension Structure

Source: `third_party_public/rank02_shiyegao_cuda_ext.py`.

This implementation is not a single fused monster kernel. It is a compact
runtime pipeline built from custom CUDA kernels plus cuBLAS Tensor Core GEMMs.

## Entry

Python side:

- Uses `torch.utils.cpp_extension.load_inline`.
- Builds one extension named like `trimul_a100_f16_v28_cap64`.
- Exposes `trimul_forward(...)`.
- `custom_kernel` passes raw tensors to the extension and does no PyTorch math.

Build flags:

- `-O3`
- `-std=c++17`
- `-arch=sm_80`
- `--use_fast_math`
- `-maxrregcount=72`

## Forward Pipeline

1. Workspace cache

   The extension keeps device-local thread-local caches:

   - `xhat`: `[B, N, N, C]` FP16 normalized input
   - `lr5`: `[5H, B, N, N]` FP16 projection/gate planes
   - `out_tmp`: `[H, B, N, N]` FP16 central contraction result
   - `out_hidden`: `[B*N*N, H]` FP16 final hidden rows
   - `mask_u8` / `mask_hist16`: optional mask compaction and profiling helpers
   - packed weights cache for six weight tensors

2. Input LayerNorm

   Custom CUDA kernel:

   - `ln_warp_affine_to_f16_kernel<128>`
   - `ln_warp_affine_to_f16_kernel<384>`
   - generic fallback for other `C`

   It computes row-wise LayerNorm in FP32 and stores FP16.

3. Weight packing

   Custom CUDA kernel:

   - `pack6_f32_to_f16_kernel`

   It packs:

   - left projection
   - right projection
   - left gate
   - right gate
   - out gate
   - final `to_out`

   into one FP16 buffer and caches by tensor pointer/version.

4. Projection GEMM

   cuBLAS:

   - helper `gemm_f16_abt`
   - computes `[5H, C] @ [rows, C].T -> [5H, rows]`
   - output is `lr5`

5. Gate + mask apply

   Custom CUDA kernels mutate `lr5` in place:

   - left plane becomes `left_proj * sigmoid(left_gate) * mask`
   - right plane becomes `right_proj * sigmoid(right_gate) * mask`
   - out-gate plane stays as logits for later

   This section is heavily specialized:

   - `int64`, `float32`, and `uint8` mask paths
   - optional mask compaction to `uint8`
   - aligned and unaligned int64 paths
   - vectorized row groups of 4
   - per-shape thread/reg-mode probes

6. Central contraction

   cuBLAS strided batched GEMM:

   - helper `gemm_strided_batched_f16_abt`
   - batch count is `H * B`
   - each batch computes `[N, N] @ [N, N].T`
   - output layout is `[H, B, N, N]`

   This confirms that direct custom central matmul is not the first target.
   cuBLAS is already strong here.

7. Hidden LayerNorm + out gate + layout conversion

   Custom CUDA kernel:

   - `ln_affine_gate_from_col_to_row_f16_kernel<128>`
   - reads central output in `[H, rows]` column-major-like layout
   - applies LayerNorm over `H`
   - multiplies `sigmoid(out_gate)`
   - writes `[rows, H]` FP16 for the final GEMM

8. Final projection

   cuBLAS:

   - helper `gemm_f16_abt`
   - computes `[rows, H] @ [dim, H].T -> [rows, dim]`
   - output is FP16, accepted by official tolerance

## Performance Meaning

The winning shape of the design is:

- Keep reductions and elementwise layout work in custom CUDA.
- Keep the three large matrix products in cuBLAS Tensor Core paths.
- Cache workspaces and packed weights so repeated timing loops do not allocate
  or repack every call.
- Specialize masks because masked benchmark/test cases are expensive enough to
  matter.

For our next code, the useful first reproduction is a smaller version of this
pipeline: custom CUDA LN, pack, gate/mask, final hidden LN/gate, with cuBLAS for
projection, central contraction, and final projection.
