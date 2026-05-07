# GPUMODE TriMul A100

Goal: build a reproducible A100 loop for GPUMODE TriMul leaderboard 496, then
move from a safe functional baseline toward shape-specialized A100 paths.

## Fixed Official Inputs

Official files are vendored under `official/` from
`gpu-mode/reference-kernels/problems/bioml/trimul`.

The local case files intentionally use `nomask:1` and `nomask:0` instead of
`True` and `False`. The official `eval.py` case parser turns numeric fields
into integers, but leaves alphabetic values as strings. A string `"False"` is
truthy in Python, so numeric mask flags keep local masked cases honest.

## Submissions

- `submissions/v00_sample.py`: official sample.
- `submissions/v01_functional_bf16.py`: no `nn.Module`, functional PyTorch,
  BF16 central einsum.
- `submissions/v02_concat_bmm_fp16.py`: one concatenated projection GEMM,
  FP16 batched central matmul.
- `submissions/v10_hf_triton_a100.py`: public Hugging Face community A100
  Triton kernel with `custom_kernel` alias added for the official evaluator.
- `submissions/v20_cuda_ext_skeleton.py`: our first inline CUDA/cuBLAS
  pipeline. It passes official tests and reproduces the rank #2 architecture at
  a smaller, not-yet-tuned level.
- `submissions/v21_cuda_ext_cache.py`: v20 plus per-device workspace reuse and
  packed FP16 weight reuse. It passes official tests and gives a modest
  geomean improvement, but does not solve the large-shape bottleneck.
- `submissions/v22_cuda_ext_warp_ln.py`: v21 plus warp-per-row input
  LayerNorm for the important `C=128/384/768` cases. It passes official tests
  and moves the custom CUDA/cuBLAS path below 10ms geomean in the best local
  A100 run.
- `submissions/v23_cuda_ext_hidden_warp_ln.py`: v22 plus warp-per-row hidden
  LayerNorm, out-gate, and `[H, rows] -> [rows, H]` conversion. It passes
  official tests and leaderboard-style recheck, landing around 4.3ms geomean
  on the local A100.
- `submissions/v24_cuda_ext_nomask_gate.py`: v23 plus a split gate path that
  skips mask loads for official float32 all-ones masks. It passes official
  tests and leaderboard-style recheck, but is a modest/noisy improvement rather
  than a new architecture step.
- `submissions/v25_cuda_ext_stage_timing.py`: v23 plus opt-in CUDA event stage
  timing. Use this as an analysis tool, not a fastest submission.
- `submissions/v26_cuda_ext_hidden_tiled.py`: v23 plus a tiled shared-memory
  hidden LayerNorm/out-gate/layout kernel. It passes official tests and
  leaderboard-style recheck, landing around 3.2ms geomean on the local A100.
- `submissions/v27_cuda_ext_stage_timing_v26.py`: v26 plus opt-in CUDA event
  stage timing. Use this as an analysis tool for the post-v26 bottlenecks.
- `submissions/v28_cuda_ext_nomask_gate_tiled.py`: v26 plus the v24-style
  float-mask nomask gate shortcut. Correct, but not a stable improvement.
- `submissions/v29_cuda_ext_hidden_tiled16.py`: v26 with a 16-row hidden tile.
  It helps large shapes but hurts small/mid shapes.
- `submissions/v30_cuda_ext_hidden_tile_dispatch.py`: v26 with tile8 for
  `N < 768` and tile16 for `N >= 768`. Correct and promising, but current
  recheck is roughly tied with v26.
- `submissions/v31_cuda_ext_tile_dispatch_nomask.py`: v30 plus the nomask gate
  shortcut. It passes official tests and leaderboard-style recheck, but is
  noisier than v32.
- `submissions/v32_cuda_ext_c384_warp_ln.py`: v30 plus warp-per-row input
  LayerNorm for all `C=384` shapes. It passes official tests and
  leaderboard-style recheck, landing around 3.0ms geomean on the local A100.
- `submissions/v33_cuda_ext_c384_warp_large_nomask.py`: v32 plus a selective
  large-shape nomask gate shortcut for float masks with `N >= 768`. It passes
  official tests, but does not beat v32.
- `submissions/v34_cuda_ext_gemm_algo_tune.py`: v32 plus env-controlled
  cuBLAS algorithm selection for projection, central, and final GEMMs. It
  passes official tests, but current sweeps do not beat v32.
- `submissions/v35_cuda_ext_cublaslt_proj.py`: v32 plus optional cuBLASLt
  projection/final GEMMs, workspace, heuristic-index controls, and an
  experimental shape-specialized auto dispatch. It passes official tests, but
  current rechecks are too noisy to promote over v32.
- `submissions/v36_cuda_ext_stage_timing_v32.py`: v32 plus opt-in CUDA event
  timing for input LN, pack, projection, gate/mask, central GEMM, hidden
  LN/gate/layout, and final projection. Use this as the current microscope,
  not as the fastest submission.
- `submissions/v37_cuda_ext_vec_gate.py`: v32 plus a vectorized H-major
  gate/mask kernel and an env fallback (`TRIMUL_V37_OLD_GATE=1`) to the old
  scalar gate path. It passes official tests and is a small same-session
  improvement over v32, but not a large enough win to close the gap by itself.
- `submissions/v38_cuda_ext_c384_reg_ln.py`: v37 plus experimental C384 input
  LayerNorm variants. `TRIMUL_V38_C384_LN_MODE=1` is one-warp register reuse,
  `TRIMUL_V38_C384_LN_MODE=2` is two-warps-per-row, and
  `TRIMUL_V38_OLD_C384_LN=1` falls back to the v37 C384 LN. It passes official
  tests, but is not promoted because the C384 LN stage did not improve.
- `submissions/v39_rank02_stage_timing.py`: analysis-only rank02 copy with
  per-call CUDA event timing. It emits `trimul_rank02_stage_v39` rows so rank02
  can be compared against v36/v37 in the same stage harness.
- `submissions/v40_cuda_ext_rank02_hidden.py`: v37 plus the rank02-style hidden
  LayerNorm/out-gate/layout kernel. It passes official tests and was the first
  native path near rank02, with leaderboard-style rechecks at 2673.060 /
  2643.726 us.
- `submissions/v41_rank01_stage_timing.py`: analysis-only rank01 Triton timing
  copy with mask inference fixed for this evaluator.
- `submissions/v42_hybrid_rank01_c128.py`: v40 fallback plus rank01 Triton path
  for `B=1, C=128, N in {512,768,1024}`. It passes official tests and is the
  current best path, with leaderboard-style rechecks at 2503.175 / 2478.318 us.
- `submissions/v43_hybrid_rank01_c128_cache.py`: v42 plus cached rank01 fp16
  weights and work buffers. It passes official tests and benchmarks well in
  long loops, but rechecks did not beat v42.
- `submissions/v44_hybrid_rank01_c128_weight_cache.py`: v42 plus rank01 fp16
  weight cache only. It is safer than v43, but still not promoted over v42.
- `submissions/v45_hybrid_rank01_late_v40.py`: rank01-first dispatch-order
  experiment. It passes correctness tests, but benchmark mode exits 112.
- `third_party_public/rank01_ttt_a100.py`: public export for A100 rank 1,
  kept for reading and comparison.
- `third_party_public/rank02_shiyegao_cuda_ext.py`: public export for A100 rank
  2, inline CUDA extension, kept for reading and optional benchmarking.

## A100 Commands

Start the existing spot A100 VM:

```bash
gpumode/trimul_a100/scripts/gcp_start_a100.sh
```

Run smoke, full test, or benchmark:

```bash
gpumode/trimul_a100/scripts/gcp_eval_submission.sh test gpumode/trimul_a100/submissions/v01_functional_bf16.py gpumode/trimul_a100/cases/smoke_cases.txt
gpumode/trimul_a100/scripts/gcp_eval_submission.sh test gpumode/trimul_a100/submissions/v01_functional_bf16.py
gpumode/trimul_a100/scripts/gcp_eval_submission.sh benchmark gpumode/trimul_a100/submissions/v01_functional_bf16.py
```

Run stage timing on benchmark shapes:

```bash
gpumode/trimul_a100/scripts/gcp_stage_timing.sh
```

Run the v39 rank02-vs-v37 stage comparison:

```bash
gpumode/trimul_a100/scripts/gcp_v39_stage_compare.sh
```

Run the v41 all-7 v40/rank02/rank01 stage comparison:

```bash
gpumode/trimul_a100/scripts/gcp_v41_stage_compare.sh
```

Run the initial baseline suite:

```bash
gpumode/trimul_a100/scripts/gcp_run_baseline_suite.sh benchmark
```

Parse logs into shape rows and append a CSV:

```bash
gpumode/trimul_a100/scripts/parse_popcorn_log.py gpumode/trimul_a100/logs/*.out --csv gpumode/trimul_a100/logs/benchmark_results.csv
```

Stop the VM when done:

```bash
gpumode/trimul_a100/scripts/gcp_stop_a100.sh
```

## First Decisions To Measure

1. Does `v01_functional_bf16.py` pass all official tests and beat the sample?
2. Does `v02_concat_bmm_fp16.py` pass tolerance on cauchy cases?
3. For each of the 7 benchmark shapes, does central `bmm` beat `einsum`?
4. Which shapes need Triton/CUDA fusion first: projection, central, or final
   hidden LN + out gate + final projection?
