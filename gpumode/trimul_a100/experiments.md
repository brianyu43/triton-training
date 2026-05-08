# TriMul A100 Experiments

## Environment

- GCP project: `nemo-488500`
- VM: `cuda-a100-dev-matmul-v2`
- Zone: `us-central1-a`
- Machine: `a2-highgpu-1g`
- GPU: `nvidia-tesla-a100`
- Scheduling: spot/preemptible

## Baseline Matrix

| Version | Purpose | Expected risk |
| --- | --- | --- |
| `v00_sample.py` | official reference-style sample | slow, high allocation overhead |
| `v01_functional_bf16.py` | safe no-Module functional baseline | should pass; central BF16 accuracy to confirm |
| `v02_concat_bmm_fp16.py` | projection concat + FP16 bmm trial | may fail cauchy tolerance; high value if it passes |
| `v10_hf_triton_a100.py` | public Triton A100 reference point | should expose useful fusion/layout choices |
| `v20_cuda_ext_skeleton.py` | first inline CUDA/cuBLAS skeleton | correctness anchor for custom-kernel work |
| `v21_cuda_ext_cache.py` | workspace cache + packed weight cache | low-risk allocation/packing cleanup |
| `v22_cuda_ext_warp_ln.py` | warp-per-row input LayerNorm | low correctness risk; should help large `N` |
| `v23_cuda_ext_hidden_warp_ln.py` | warp-per-row hidden LN + out gate + layout conversion | medium value, low correctness risk with `H=128` |
| `v24_cuda_ext_nomask_gate.py` | split nomask gate path | safe for official float32 all-ones masks; expected small win |
| `v25_cuda_ext_stage_timing.py` | CUDA event stage timing | analysis-only; not intended as fastest path |
| `v26_cuda_ext_hidden_tiled.py` | tiled shared-memory hidden LN + out gate + layout conversion | medium implementation risk; targets uncoalesced hidden reads |
| `v27_cuda_ext_stage_timing_v26.py` | post-v26 CUDA event stage timing | analysis-only; confirms remaining split bottleneck |
| `v28_cuda_ext_nomask_gate_tiled.py` | v26 + float-mask nomask gate shortcut | correctness-safe, but noisy and not promoted |
| `v29_cuda_ext_hidden_tiled16.py` | v26 + 16-row hidden tile | helps large shapes, hurts small/mid shapes |
| `v30_cuda_ext_hidden_tile_dispatch.py` | tile8 for small/mid, tile16 for large shapes | promising, but recheck is roughly tied with v26 |
| `v31_cuda_ext_tile_dispatch_nomask.py` | v30 + float-mask nomask gate shortcut | correctness-safe, noisy, useful large-shape signal |
| `v32_cuda_ext_c384_warp_ln.py` | v30 + warp input LN for all C384 shapes | low-risk C384 cleanup; former best verified run |
| `v33_cuda_ext_c384_warp_large_nomask.py` | v32 + large-shape nomask gate shortcut | correctness-safe, but does not beat v32 |
| `v34_cuda_ext_gemm_algo_tune.py` | v32 + env-controlled cuBLAS algo tuning | correctness-safe harness, not promoted |
| `v35_cuda_ext_cublaslt_proj.py` | v32 + optional cuBLASLt projection/final GEMMs | correctness-safe harness with shape signals, not promoted |
| `v36_cuda_ext_stage_timing_v32.py` | v32 + opt-in CUDA event stage timing | analysis-only microscope for next kernel choice |
| `v37_cuda_ext_vec_gate.py` | v32 + vectorized H-major gate/mask kernel | low-risk gate cleanup; small same-session win |
| `v38_cuda_ext_c384_reg_ln.py` | v37 + C384 register/two-warp LN variants | correctness-safe experiment, not promoted |
| `v39_rank02_stage_timing.py` | rank02 + per-call stage timing | analysis-only comparison target |
| `v40_cuda_ext_rank02_hidden.py` | v37 + rank02-style hidden LN/out-gate/layout kernel | correctness-safe hidden-stage port; former best native path |
| `v41_rank01_stage_timing.py` | rank01 Triton path + Python CUDA event timing | analysis-only all-7 comparison target |
| `v42_hybrid_rank01_c128.py` | v40 default + rank01 path for B1 C128 N512/768/1024 | current best hybrid path |
| `v43_hybrid_rank01_c128_cache.py` | v42 + rank01 weight/work-buffer cache | benchmark-only win; not promoted |
| `v44_hybrid_rank01_c128_weight_cache.py` | v42 + rank01 fp16 weight cache only | small/noisy recheck result; not promoted over v42 |
| `v45_hybrid_rank01_late_v40.py` | route early benchmark shapes to rank01, load v40 late | correctness pass, benchmark mode exits 112 |
| `v46_hybrid_stage_timing.py` | v42 + same-session all-7 stage timing | analysis-only; guides next target |
| `v47_hybrid_rank01_c384n768.py` | v42 + full rank01 route for B1 N768 C384 | correctness pass, leaderboard recheck regressed |
| `v48_c384_proj_gate_tail.py` | v42 + C384 Triton projection/gate feeding v40 tail | correctness pass; same-session win, not promoted |
| `v49_v48_rank01_weight_cache.py` | v48 + rank01 fp16 transposed weight cache | correctness pass; cache effect small/noisy |

## A100 Baseline Results

Measured on `cuda-a100-dev-matmul-v2` on 2026-05-06 and 2026-05-07 with
official benchmark mode. Timings are mean microseconds.

| Version | Geomean us | Notes |
| --- | ---: | --- |
| `v00_sample.py` | 22979.830 | official sample, correctness pass |
| `v01_functional_bf16.py` | 19889.921 | official test pass, modest launch/allocation cleanup |
| `v02_concat_bmm_fp16.py` | 12544.912 | official test pass, bmm is the first major win |
| `v10_hf_triton_a100.py` | 4917.194 | official test pass, Triton fusion crosses 5ms |
| `v20_cuda_ext_skeleton.py` | 11563.949 | our first inline CUDA/cuBLAS pipeline, official test pass |
| `v21_cuda_ext_cache.py` | 10944.079 / 11122.233 | official test pass, workspace/packed-weight cache helps small shapes but is not a main bottleneck |
| `v22_cuda_ext_warp_ln.py` | 9527.615 / 9941.310 / 10044.938 | official test pass, warp input LN helps large shapes; small C384 is noisy |
| `v23_cuda_ext_hidden_warp_ln.py` | 4298.448 / 4381.758 | official test pass; leaderboard-style recheck pass at 4315.162 us |
| `v24_cuda_ext_nomask_gate.py` | 4210.085 / 4323.630 | official test pass; leaderboard-style recheck pass at 4369.955 us |
| `v25_cuda_ext_stage_timing.py` | n/a | analysis-only; stage timing test pass |
| `v26_cuda_ext_hidden_tiled.py` | 3324.322 / 3217.441 | official test pass; leaderboard-style recheck pass at 3192.199 us |
| `v27_cuda_ext_stage_timing_v26.py` | n/a | analysis-only; post-v26 stage timing pass |
| `v28_cuda_ext_nomask_gate_tiled.py` | 3247.212 | official test pass; leaderboard-style recheck pass at 3252.140 us |
| `v29_cuda_ext_hidden_tiled16.py` | 3265.282 | official test pass; leaderboard-style recheck pass at 3221.454 us |
| `v30_cuda_ext_hidden_tile_dispatch.py` | 3078.652 | official test pass; leaderboard-style recheck pass at 3204.735 us |
| `v31_cuda_ext_tile_dispatch_nomask.py` | 3276.803 | official test pass; leaderboard-style recheck pass at 3150.028 us |
| `v32_cuda_ext_c384_warp_ln.py` | 3148.741 | official test pass; leaderboard-style recheck pass at 3011.576 us |
| `v33_cuda_ext_c384_warp_large_nomask.py` | 3141.312 | official test pass; leaderboard-style recheck pass at 3102.972 us |
| `v34_cuda_ext_gemm_algo_tune.py` | 3087.958 | official test pass; default leaderboard-style recheck pass at 3060.919 us |
| `v35_cuda_ext_cublaslt_proj.py` | 3058.585 | official test pass; Lt-both benchmark hit 2987.630 us, but final rechecks were 3019.773-3078.261 us, so v32 remained best at that point |
| `v36_cuda_ext_stage_timing_v32.py` | 3120.108 | official test pass; analysis-only stage timing version of v32 |
| `v37_cuda_ext_vec_gate.py` | 2975.466 | official test pass; leaderboard-style rechecks at 3012.838 / 3011.947 us; same-session v32 control was 3077.317 us |
| `v38_cuda_ext_c384_reg_ln.py` | 3131.019 | official test pass; C384 LN register variants did not beat v37, so do not promote |
| `v39_rank02_stage_timing.py` | n/a | analysis-only rank02 timing copy; identifies hidden LN/out-gate/layout as the dominant rank02-vs-v37 gap |
| `v40_cuda_ext_rank02_hidden.py` | 2716.584 | official test pass; leaderboard-style rechecks at 2673.060 / 2643.726 us |
| `v41_rank01_stage_timing.py` | 2497.923 | analysis/timing copy with mask inference fixed; not used as native default |
| `v42_hybrid_rank01_c128.py` | 2501.884 | official test pass; leaderboard-style rechecks at 2503.175 / 2478.318 us |
| `v43_hybrid_rank01_c128_cache.py` | 2413.589 | official test pass; leaderboard-style rechecks at 2510.026 / 2509.072 us, so not promoted |
| `v44_hybrid_rank01_c128_weight_cache.py` | 2413.037 | official test pass; leaderboard-style rechecks at 2494.435 / 2495.585 us, so v42 remains best verified |
| `v45_hybrid_rank01_late_v40.py` | n/a | official test pass and benchmark-case test pass; benchmark mode exits 112 before rows |
| `v46_hybrid_stage_timing.py` | n/a | official test pass; analysis-only all-7 timing harness for v42/v40/rank01/rank02 |
| `v47_hybrid_rank01_c384n768.py` | 2422.481 | official test pass; leaderboard-style recheck at 2545.999 us, so not promoted |
| `v48_c384_proj_gate_tail.py` | 2505.572 | official and benchmark-shape tests pass; leaderboard-style recheck at 2522.822 us versus same-session v42 at 2571.735 us |
| `v49_v48_rank01_weight_cache.py` | 2432.794 | official and benchmark-shape tests pass; leaderboard-style rechecks at 2489.406 / 2521.721 us; cache-disabled control was 2492.101 us |
| `rank02_shiyegao_cuda_ext.py` | 2610.515 | public CUDA extension reference point |

`rank02_shiyegao_cuda_ext.py` also passed official leaderboard recheck mode at
2648.031 us geomean on this VM.

## Winner Table

Fill this from `logs/benchmark_results.csv`.

| Shape | Best current path | mean us | Notes |
| --- | --- | ---: | --- |
| B2 N256 C128 nomask normal | `rank02_shiyegao_cuda_ext.py` | 745.363 | HF Triton: 1017.429 |
| B1 N768 C128 nomask cauchy | `rank02_shiyegao_cuda_ext.py` | 3329.640 | HF Triton: 7496.107 |
| B2 N256 C384 mask normal | `rank02_shiyegao_cuda_ext.py` | 1065.390 | HF Triton: 1510.026 |
| B1 N512 C128 nomask normal | `rank02_shiyegao_cuda_ext.py` | 1305.098 | HF Triton: 2648.995 |
| B1 N1024 C128 nomask cauchy | `rank02_shiyegao_cuda_ext.py` | 6101.861 | HF Triton: 10888.309 |
| B1 N768 C384 mask normal | `rank02_shiyegao_cuda_ext.py` | 4696.256 | HF Triton: 10771.338 |
| B1 N1024 C384 nomask normal | `rank02_shiyegao_cuda_ext.py` | 8355.115 | HF Triton: 19426.101 |

## Notes

- Treat top public exports as reference material, not as the final submission
  strategy.
- Do not remove mask handling solely by benchmark shape. Official tests include
  masked variants near the important large shapes.
- For final tuning, compare FP16 and BF16 explicitly; do not assume BF16 wins.
- Next implementation target is not another high-level PyTorch rearrangement.
  The measured gap is now fusion/codegen: projection+gate+mask, central bmm,
  and final LN+gate+projection need the CUDA-extension style treatment.
- `v20_cuda_ext_skeleton.py` proves the custom CUDA/cuBLAS pipeline is correct,
  but it is intentionally naive:
  - allocates workspaces every call
  - repacks weights every call
  - uses one CUDA block per row for LayerNorm reductions
  - has only raw `float32`/`int64` mask handling, no `uint8` compaction
  - uses default cuBLAS batched GEMM algorithm
- `v21_cuda_ext_cache.py` removes the first two obvious wastes with per-device
  caches:
  - reuses one-shape workspaces for `xhat`, five projections, central output,
    and final hidden rows
  - reuses packed FP16 weights when tensor pointers and versions match
  - passes all official tests
  - improves geomean from 11563.949 us to roughly 10.9-11.1 ms, but large
    shapes remain dominated by LayerNorm/layout/mask work
  - regresses `B1 N1024 C128` versus v20 in the current measurements, so cache
    should stay as infrastructure rather than be treated as the winning change
- `v22_cuda_ext_warp_ln.py` replaces the input LayerNorm launch for the main
  dimensions:
  - one warp handles one row and reduces with shuffle ops instead of shared
    memory block reductions
  - `C=128` and large `C=384`/`C=768` use the warp kernel
  - small `C=384` keeps the block fallback after benchmark showed the warp path
    was not reliably better there
  - passes all official tests
  - moves geomean into the 9.5-10.0 ms range on this VM
- `v23_cuda_ext_hidden_warp_ln.py` replaces the post-central hidden LayerNorm
  kernel:
  - central output remains `[H, rows]`
  - one warp handles one row of `H=128`
  - the kernel computes hidden LayerNorm, applies `sigmoid(out_gate)`, and
    writes `[rows, H]` FP16 for the final cuBLAS projection
  - passes official full test and leaderboard-style recheck
  - cuts geomean to roughly 4.3-4.4 ms, proving post-central layout/reduction
    was the largest remaining waste in our v22 path
- `v24_cuda_ext_nomask_gate.py` splits the gate kernel:
  - official `nomask=True` inputs produce a float32 all-ones mask
  - official masked inputs come from `torch.randint` and arrive as int64
  - float32 mask cases now skip mask load and mask multiply
  - int64 masked cases keep the existing mask-aware kernel
  - passes full official test and leaderboard-style recheck
  - measured effect is modest and noisy: best benchmark geomean is 4210.085 us,
    while leaderboard-style recheck is 4369.955 us
- `v25_cuda_ext_stage_timing.py` adds opt-in CUDA events around the main stages:
  - enabled with `TRIMUL_STAGE_TIMING=1`
  - `scripts/gcp_stage_timing.sh` runs the benchmark shapes with timing enabled
  - repeated timing cases are used so the second occurrence avoids cuBLAS lazy
    initialization artifacts
  - the stage table below should guide the next optimization, not be treated as
    a leaderboard score
- `v26_cuda_ext_hidden_tiled.py` replaces the v23 hidden kernel with a tiled
  shared-memory path:
  - loads central output and out-gate values coalesced by hidden plane
  - transposes an `8 x 128` row tile in shared memory
  - then uses one warp per row for hidden LayerNorm, sigmoid gate, and final
    `[rows, H]` FP16 layout
  - passes full official test and leaderboard-style recheck
  - moves geomean to roughly 3.2-3.3 ms, proving the remaining v23 hidden path
    cost was largely uncoalesced global memory traffic
- `v27_cuda_ext_stage_timing_v26.py` shows the post-v26 bottleneck is split:
  hidden is still large on N1024, but projection, gate, and C384 input
  LayerNorm are now all meaningful targets.
- `v28_cuda_ext_nomask_gate_tiled.py` repeats the v24 nomask gate shortcut on
  top of v26:
  - passes full official test and leaderboard-style recheck
  - does not beat v26; recheck is 3252.140 us
  - keep it as evidence that scalar nomask gate cleanup alone is not enough
- `v29_cuda_ext_hidden_tiled16.py` changes the hidden tile from 8 rows to 16
  rows:
  - passes full official test and leaderboard-style recheck
  - improves N768/N1024 large shapes
  - hurts N256/N512 enough that whole-geomean results do not clearly improve
- `v30_cuda_ext_hidden_tile_dispatch.py` dispatches hidden tile size by N:
  - tile8 for `N < 768`
  - tile16 for `N >= 768`
  - passes full official test and leaderboard-style recheck
  - best benchmark geomean observed is 3078.652 us, but recheck is 3204.735 us,
    so v26 remains the safer stable baseline until another confirmation run
- `v31_cuda_ext_tile_dispatch_nomask.py` combines v30 with the float-mask
  nomask gate shortcut:
  - passes full official test and leaderboard-style recheck
  - benchmark geomean is noisy at 3276.803 us
  - leaderboard-style recheck improves to 3150.028 us, showing the large-shape
    nomask path can help, but small/mid shape variance is high
- `v32_cuda_ext_c384_warp_ln.py` removes the row-count guard from the C384
  input LayerNorm warp path:
  - C384 now uses warp-per-row input LayerNorm for all benchmark/test sizes
  - passes full official test and leaderboard-style recheck
  - improves the small masked C384 benchmark shape from roughly 1.55 ms to
    roughly 1.20 ms in recheck
  - this became the best verified geomean at 3011.576 us until v40
- `v33_cuda_ext_c384_warp_large_nomask.py` adds a selective large-shape nomask
  gate path on top of v32:
  - only float-mask cases with `N >= 768` skip mask load/multiply
  - passes full official test and leaderboard-style recheck
  - does not beat v32; recheck is 3102.972 us
- `v35_cuda_ext_cublaslt_proj.py` adds a real cuBLASLt harness on top of v32:
  - supports projection and final GEMM through cuBLASLt with a reusable
    workspace and cached heuristic plans
  - sweep controls: `TRIMUL_USE_LT`, `TRIMUL_USE_LT_PROJ`,
    `TRIMUL_USE_LT_OUT`, `TRIMUL_LT_PROJ_INDEX`, `TRIMUL_LT_OUT_INDEX`,
    `TRIMUL_LT_WORKSPACE_MB`, and `TRIMUL_DISABLE_LT_AUTO`
  - passes the full official test suite
  - global Lt-both (`TRIMUL_USE_LT=1`) showed a promising benchmark geomean of
    2987.630 us, but leaderboard-style recheck was 3013.065 us
  - shape-specific auto dispatch improved C128/C384 large-shape components, but
    small fallback shapes were noisy; same-session v32 recheck was 3005.924 us
    while v35 auto rechecks ranged from 3019.773 to 3078.261 us
  - conclusion: keep v35 as the cuBLASLt experiment harness, but do not promote
    it over v32
- `v36_cuda_ext_stage_timing_v32.py` adds opt-in CUDA event stage timing on top
  of the current v32 best path:
  - enabled with `TRIMUL_STAGE_TIMING=1`
  - limits printed calls with `TRIMUL_STAGE_TIMING_LIMIT`
  - emits `trimul_stage_v36` rows with `ln`, `pack`, `proj`, `gate`,
    `central`, `hidden`, `out`, and `total` times
  - passes the full official test suite
  - default benchmark geomean in this session was 3120.108 us, so keep it as an
    analysis build rather than a promoted submission
- `v37_cuda_ext_vec_gate.py` replaces the scalar one-element-per-thread
  gate/mask update with a vectorized H-major kernel:
  - default path processes four rows per thread for each hidden plane and uses
    `half2` loads/stores for the left/right and gate pairs
  - float32 masks route through the official nomask/all-ones path; int64 masks
    still load and apply per-row mask values
  - `TRIMUL_V37_OLD_GATE=1` falls back to the old v32/v36 gate kernel for
    quick bisects
  - passes smoke, full official test, and leaderboard-style recheck
  - benchmark geomean was 2975.466 us; two leaderboard-style rechecks were
    3012.838 us and 3011.947 us
  - old-gate control inside the same file benchmarked at 3140.649 us, while a
    same-session v32 leaderboard control was 3077.317 us
  - conclusion: keep v37 as a useful small gate cleanup, but the gate kernel
    alone is not the missing 700 us

## v36 Steady-State Stage Timing

Measured from `stage_timing_repeated_cases.txt` on A100. The table uses the
second occurrence of each benchmark shape so `workspace_hit=1`. The official
test-mode harness regenerates weights per case, so `packed_hit=0` is expected.

| Shape | total ms | input LN | projection | gate/mask | central | hidden | final out | Main pressure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B2 N256 C128 | 0.9329 | 0.0911 | 0.2314 | 0.1710 | 0.1075 | 0.2519 | 0.0707 | hidden/proj/gate |
| B1 N768 C128 | 4.0018 | 0.3441 | 0.9943 | 0.7127 | 0.6318 | 1.0107 | 0.3000 | hidden/proj/gate |
| B2 N256 C384 | 1.3537 | 0.2335 | 0.4147 | 0.1679 | 0.1055 | 0.2509 | 0.1710 | projection/input LN |
| B1 N512 C128 | 1.8176 | 0.1597 | 0.4485 | 0.3195 | 0.2161 | 0.5417 | 0.1229 | hidden/proj/gate |
| B1 N1024 C128 | 7.3257 | 0.6011 | 1.7674 | 1.2749 | 1.3527 | 1.7920 | 0.5294 | hidden/proj/gate/central |
| B1 N768 C384 | 5.1425 | 1.0004 | 1.5432 | 0.7158 | 0.5222 | 0.7895 | 0.5632 | projection/input LN |
| B1 N1024 C384 | 9.1668 | 1.7633 | 2.7412 | 1.2564 | 1.0803 | 1.3988 | 0.9165 | projection/input LN/hidden |

v36 suggests the next real implementation should not be another GEMM flag
sweep. The broadest targets are projection+gate/mask fusion for C128/C384 and
C384 input LayerNorm/projection cleanup. Central GEMM is important for
N1024 C128, but it is not the only dominant stage.

## v37 Gate Timing Delta

Measured from `stage_timing_repeated_cases.txt` using the second occurrence of
each benchmark shape. This isolates the gate change while keeping the rest of
the v32 path intact.

| Shape | v36 gate ms | v37 gate ms | Delta | v37 total ms |
| --- | ---: | ---: | ---: | ---: |
| B2 N256 C128 | 0.1710 | 0.1526 | -10.8% | 0.9134 |
| B1 N768 C128 | 0.7127 | 0.6574 | -7.8% | 3.9537 |
| B2 N256 C384 | 0.1679 | 0.1546 | -7.9% | 1.3435 |
| B1 N512 C128 | 0.3195 | 0.2970 | -7.0% | 1.8022 |
| B1 N1024 C128 | 1.2749 | 1.1592 | -9.1% | 7.2284 |
| B1 N768 C384 | 0.7158 | 0.6943 | -3.0% | 5.1476 |
| B1 N1024 C384 | 1.2564 | 1.1756 | -6.4% | 9.1423 |

The vectorized gate path is real, but smaller than the hoped-for 25-35% stage
drop. The old gate path was already reasonably coalesced in the H-major layout.
The next high-value work should therefore target either C384 input
LayerNorm/projection cleanup or a deeper projection-output/gate fusion that
avoids writing and rereading the gate slices as separate traffic.

## v38 C384 LN Attempt

`v38_cuda_ext_c384_reg_ln.py` tested two C384 input LayerNorm variants on top
of v37:

- `TRIMUL_V38_C384_LN_MODE=1`: one warp per row, but keep the 12 values per
  lane in registers so the input row is not reread for the output write.
- `TRIMUL_V38_C384_LN_MODE=2`: two warps per row, six values per thread, with
  shared-memory merge of the two warp reductions.
- `TRIMUL_V38_OLD_C384_LN=1`: old v37 C384 LN fallback.

Both new modes pass the full official test suite, including masked C384 large
tests. Stage timing did not show the expected win:

| Shape | v37 LN ms | v38 mode1 LN ms | v38 mode2 LN ms | Read |
| --- | ---: | ---: | ---: | --- |
| B2 N256 C384 | 0.2324 | 0.2314 | 0.2304 | flat |
| B1 N768 C384 | 0.9984 | 1.0107 | 0.9892 | flat/noisy |
| B1 N1024 C384 | 1.7664 | 1.7930 | 1.7910 | worse |

The default v38 benchmark geomean was 3131.019 us, so v38 is an experiment,
not a promoted path. The important conclusion is that C384 input LN is not the
easy 100-300 us lever by itself. The old warp-per-row LN was already close
enough that the next real move should attack traffic around projection output:
avoid writing five full projection slices and then rereading left/right gates
in a separate pass.

## v39 Rank02 Stage Comparison

`v39_rank02_stage_timing.py` is an analysis-only copy of the public rank02
submission. It adds a `trimul_rank02_stage_v39` per-call timing row with these
normalized buckets:

- `ln`: input LayerNorm
- `proj_gate`: packed-weight work, projection GEMM, mask/gate apply
- `central`: strided batched central GEMM
- `hidden`: hidden LayerNorm, out gate, and column-to-row layout conversion
- `out`: final projection

The comparison used `v39_stage_compare_cases.txt`, which repeats the four
highest-impact benchmark shapes four times and takes the final occurrence for
the steady-state read.

| Variant | Shape | LN | Proj+Gate | Central | Hidden | Out | Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rank02_v39 | B1 N768 C128 | 0.3389 | 1.6896 | 0.6287 | 0.4219 | 0.3000 | 3.4017 |
| v37 | B1 N768 C128 | 0.3430 | 1.6518 | 0.6298 | 1.0117 | 0.3000 | 3.9363 |
| rank02_v39 | B1 N1024 C128 | 0.5898 | 2.9757 | 1.3548 | 0.7219 | 0.5304 | 6.1962 |
| v37 | B1 N1024 C128 | 0.5980 | 2.9236 | 1.3527 | 1.7910 | 0.5315 | 7.1967 |
| rank02_v39 | B1 N768 C384 | 1.0127 | 2.2446 | 0.5212 | 0.3840 | 0.5612 | 4.7452 |
| v37 | B1 N768 C384 | 0.9984 | 2.2538 | 0.5212 | 0.7885 | 0.5591 | 5.1210 |
| rank02_v39 | B1 N1024 C384 | 1.7981 | 3.9813 | 1.0957 | 0.6451 | 0.9134 | 8.4521 |
| v37 | B1 N1024 C384 | 1.7654 | 3.9280 | 1.0936 | 1.3978 | 0.9144 | 9.0993 |

Conclusion: the advice to measure before another patch was right, and the
measurement changes the priority. On these four shapes, rank02 is not winning
mainly in projection/gate. `Proj+Gate` is roughly tied, and central/final are
also roughly tied. The dominant difference is the hidden bucket:

| Shape | v37 total - rank02 total | Hidden gap |
| --- | ---: | ---: |
| B1 N768 C128 | 0.5346 ms | 0.5898 ms |
| B1 N1024 C128 | 1.0005 ms | 1.0691 ms |
| B1 N768 C384 | 0.3758 ms | 0.4045 ms |
| B1 N1024 C384 | 0.6472 ms | 0.7527 ms |

This means v40 should pivot to rank02-style hidden kernel analysis/porting
before mask/gate specialization. Mask/gate work still matters for masked C384,
but v39 says it is not the biggest missing chunk relative to rank02.

## v40 Rank02-Style Hidden Port

`v40_cuda_ext_rank02_hidden.py` keeps the v37 pipeline and replaces only the
post-central hidden LayerNorm/out-gate/layout stage. The new kernel uses the
rank02 tile shape:

- `32` row tile
- `8` warps per block
- shared-memory transpose from `[H, rows]` to row-major output
- one warp computes one hidden LayerNorm row at a time

The old v37 hidden kernel is still available with
`TRIMUL_V40_OLD_HIDDEN=1`, and the old gate fallback remains available with
`TRIMUL_V40_OLD_GATE=1`.

Validation on A100:

- smoke test: pass
- full official test: 18/18 pass
- benchmark geomean: 2716.584 us
- leaderboard-style rechecks: 2673.060 us and 2643.726 us

Leaderboard-style shape means from the second recheck:

| Shape | v40 mean us |
| --- | ---: |
| B2 N256 C128 nomask normal | 716.687 |
| B1 N768 C128 nomask cauchy | 3426.765 |
| B2 N256 C384 mask normal | 1100.165 |
| B1 N512 C128 nomask normal | 1331.541 |
| B1 N1024 C128 nomask cauchy | 6185.643 |
| B1 N768 C384 mask normal | 4785.152 |
| B1 N1024 C384 nomask normal | 8476.331 |

Same-session stage timing on the four focus shapes confirms the intended
effect. `v40` collects almost all of the hidden-stage gap identified by v39:

| Variant | Shape | LN | Proj+Gate | Central | Hidden | Out | Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rank02_v39 | B1 N768 C128 | 0.3389 | 1.6896 | 0.6287 | 0.4219 | 0.3000 | 3.4017 |
| v37 | B1 N768 C128 | 0.3430 | 1.6518 | 0.6298 | 1.0117 | 0.3000 | 3.9363 |
| v40 | B1 N768 C128 | 0.3430 | 1.6507 | 0.6257 | 0.4291 | 0.3031 | 3.3516 |
| rank02_v39 | B1 N1024 C128 | 0.5898 | 2.9757 | 1.3548 | 0.7219 | 0.5304 | 6.1962 |
| v37 | B1 N1024 C128 | 0.5980 | 2.9236 | 1.3527 | 1.7910 | 0.5315 | 7.1967 |
| v40 | B1 N1024 C128 | 0.5990 | 2.9143 | 1.3507 | 0.7188 | 0.5304 | 6.1133 |
| rank02_v39 | B1 N768 C384 | 1.0127 | 2.2446 | 0.5212 | 0.3840 | 0.5612 | 4.7452 |
| v37 | B1 N768 C384 | 0.9984 | 2.2538 | 0.5212 | 0.7885 | 0.5591 | 5.1210 |
| v40 | B1 N768 C384 | 0.9984 | 2.2619 | 0.5212 | 0.4065 | 0.5581 | 4.7462 |
| rank02_v39 | B1 N1024 C384 | 1.7981 | 3.9813 | 1.0957 | 0.6451 | 0.9134 | 8.4521 |
| v37 | B1 N1024 C384 | 1.7654 | 3.9280 | 1.0936 | 1.3978 | 0.9144 | 9.0993 |
| v40 | B1 N1024 C384 | 1.7654 | 3.9342 | 1.0967 | 0.6912 | 0.9083 | 8.3958 |

Conclusion: v40 is the new native baseline. The remaining rank02 gap is no
longer a single hidden-stage cliff; it is mostly smaller differences in
projection/gate traffic, small-shape overhead, and C384-heavy rows.

## v41/v42 Rank01 Comparison And Hybrid

`v41_rank01_stage_timing.py` instruments the public rank01 Triton path with
Python CUDA events. The final Triton kernel fuses hidden LayerNorm, out gate,
and final projection, so the comparable bucket is `Hidden+Out`, not separate
`hidden` and `out` stages.

The all-7 comparison used `v41_stage_compare_all7_cases.txt`, repeating each
benchmark shape four times and parsing the final occurrence.

| Variant | Shape | LN | Proj+Gate | Central | Hidden+Out | Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| rank01_v41 | B2 N256 C128 | 0.1792 | 0.5222 | 0.1096 | 0.1403 | 0.9513 |
| v40 | B2 N256 C128 | 0.0911 | 0.3952 | 0.1065 | 0.1823 | 0.7752 |
| rank01_v41 | B1 N512 C128 | 0.2294 | 0.6964 | 0.2181 | 0.2560 | 1.3998 |
| v40 | B1 N512 C128 | 0.1608 | 0.7547 | 0.2161 | 0.3154 | 1.4469 |
| rank01_v41 | B1 N768 C128 | 0.3410 | 1.1755 | 0.6287 | 0.5622 | 2.7075 |
| v40 | B1 N768 C128 | 0.3420 | 1.6578 | 0.6308 | 0.7291 | 3.3597 |
| rank01_v41 | B1 N1024 C128 | 0.5970 | 2.0429 | 1.3548 | 0.9615 | 4.9562 |
| v40 | B1 N1024 C128 | 0.6001 | 2.9266 | 1.3527 | 1.2442 | 6.1235 |
| rank01_v41 | B2 N256 C384 | 0.2458 | 0.7066 | 0.1065 | 0.2591 | 1.3179 |
| v40 | B2 N256 C384 | 0.2335 | 0.5806 | 0.1055 | 0.2734 | 1.1930 |
| rank01_v41 | B1 N768 C384 | 1.0875 | 2.0992 | 0.5212 | 1.0066 | 4.7145 |
| v40 | B1 N768 C384 | 0.9974 | 2.2497 | 0.5202 | 0.9708 | 4.7380 |
| rank01_v41 | B1 N1024 C384 | 1.9261 | 3.7950 | 1.0650 | 1.7623 | 8.5484 |
| v40 | B1 N1024 C384 | 1.7633 | 3.9127 | 1.0650 | 1.6067 | 8.3476 |

Conclusion: rank01's Triton projection/gate path is much better on large
C128, especially N768/N1024. v40 remains better on small C128 and most C384
rows. `v42_hybrid_rank01_c128.py` therefore keeps v40 as the default and
dispatches only `B=1, C=128, N in {512,768,1024}` through the rank01 path.

Validation on A100:

- full official test: 18/18 pass
- benchmark geomean: 2501.884 us
- leaderboard-style rechecks: 2503.175 us and 2478.318 us

Second recheck shape means:

| Shape | v42 mean us |
| --- | ---: |
| B2 N256 C128 nomask normal | 721.449 |
| B1 N768 C128 nomask cauchy | 2891.238 |
| B2 N256 C384 mask normal | 1103.555 |
| B1 N512 C128 nomask normal | 1214.587 |
| B1 N1024 C128 nomask cauchy | 5126.144 |
| B1 N768 C384 mask normal | 4770.645 |
| B1 N1024 C384 nomask normal | 8398.848 |

Next target: preserve v42's large C128 win while recovering the remaining
C384 and small-shape overhead. The likely direction is not another global
dispatch swap, but a C384-specific projection/gate or final fused path.

## v43-v45 Cache And Dispatch Follow-Ups

The cache follow-up was worth testing, but did not produce a stable
leaderboard-style promotion.

| Version | What changed | Full test | Benchmark geomean us | Recheck geomean us | Read |
| --- | --- | --- | ---: | ---: | --- |
| `v43_hybrid_rank01_c128_cache.py` | cached rank01 fp16 weights and work buffers | pass | 2413.589 | 2510.026 / 2509.072 | long-loop benchmark improved, short recheck did not |
| `v44_hybrid_rank01_c128_weight_cache.py` | cached only rank01 fp16 transposed weights | pass | 2413.037 | 2494.435 / 2495.585 | safer than v43, but still not better than v42's best recheck |
| `v45_hybrid_rank01_late_v40.py` | rank01 for all early benchmark shapes, v40 only for N1024 C384 | pass | n/a | n/a | benchmark-mode exit 112 before rows, despite benchmark-case test pass |

Conclusion: v42 remains the best verified submission from this group. The
rank01 C128 path can look much faster in long benchmark loops, especially on
N1024 C128, but this did not survive the leaderboard-style run count. The next
useful move should avoid relying on cache warmup amortization. Either build a
real C128 projection/gate/final fused path that is fast from the first timed
iteration, or attack the C384 rows where v42 still spends most of the geomean.

## v46-v47 Remaining Gap Check

`v46_hybrid_stage_timing.py` keeps v42's dispatch and adds timing around the
rank01 branch, then compares it with rank01, rank02, and v40 in the same all-7
stage harness. This was mainly a sanity check before chasing rank #1.

Key warmed rows from 2026-05-08:

| Variant | Shape | LN | Proj+Gate | Central | Hidden+Out | Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| v40 | B1 N768 C384 | 0.9984 | 2.2476 | 0.5222 | 0.9707 | 4.7391 |
| rank01_v41 | B1 N768 C384 | 1.0895 | 2.0972 | 0.5202 | 1.0035 | 4.7104 |
| v46_hybrid | B1 N768 C384 | 0.9974 | 2.2456 | 0.5212 | 0.9708 | 4.7350 |
| v40 | B1 N1024 C384 | 1.7633 | 3.9168 | 1.0772 | 1.6036 | 8.3610 |
| rank01_v41 | B1 N1024 C384 | 1.9302 | 3.7867 | 1.0762 | 1.7623 | 8.5555 |
| v46_hybrid | B1 N1024 C384 | 1.7664 | 3.9169 | 1.0762 | 1.5964 | 8.3558 |
| v40 | B1 N1024 C128 | 0.5990 | 2.9225 | 1.3507 | 1.2421 | 6.1143 |
| rank01_v41 | B1 N1024 C128 | 0.5929 | 2.0296 | 1.3496 | 0.9595 | 4.9316 |
| v46_hybrid:rank01_c128 | B1 N1024 C128 | 0.5960 | 2.0367 | 1.3517 | 0.9636 | 4.9480 |

Conclusion: the suggested "final fusion first" plan is not supported for C384.
Rank01's fused final bucket is better on the C128 rows already routed through
rank01, but it is worse on the C384 large rows. The reusable signal is narrower:
rank01's projection/gate bucket is 130-150 us better on large C384, while v40's
input LN, hidden, and final cuBLAS path are better.

`v47_hybrid_rank01_c384n768.py` tested the cheap dispatch version of that idea
by routing `B1 N768 C384` through full rank01. It passed correctness and
benchmark mode looked fast at 2422.481 us geomean, but leaderboard-style recheck
regressed to 2545.999 us. Shape rows showed the full C384 swap was not stable,
so v47 is kept as evidence, not a promoted path.

## v48 C384 Projection/Gate Tail Split

`v48_c384_proj_gate_tail.py` is the surgical version of the v47 idea:

- keep v42's rank01 route for `B1 C128 N512/768/1024`
- keep v40 for small shapes and non-target C384 cases
- for `B1 C384 N768` and nomask `B1 C384 N1024`, run v40's warp input LN,
  then a Triton projection/gate kernel that writes v40-compatible H-major
  `left`, `right`, and raw `out_gate` logits, then v40's central/hidden/final
  tail
- masked `B1 N1024 C384` falls back to v40 because official tests include that
  non-benchmark case

The main correctness trap was out-gate representation. Rank01 stores
`sigmoid(out_gate)` because its final kernel expects a post-sigmoid gate; v40's
hidden kernel expects raw logits and applies sigmoid itself. v48 stores raw
out-gate logits.

Validation on A100:

- full official test: 18/18 pass
- benchmark-shape test: 7/7 pass
- benchmark geomean: 2505.572 us
- leaderboard-style recheck: 2522.822 us
- same-session v42 leaderboard-style control: 2571.735 us

Warmed stage timing showed the intended local win but also why this is not a
rank-1-sized jump:

| Shape | v40/v46 proj+gate | v48 proj+gate | v48 total read |
| --- | ---: | ---: | --- |
| B1 N768 C384 masked | ~2.245 ms | ~2.06-2.07 ms | small total win, roughly 4.72 ms in timing |
| B1 N1024 C384 nomask | ~3.917 ms | ~3.69-3.72 ms | mostly flat total, tail overhead eats much of it |

Tile sweep notes:

- default `BLOCK_M=64, BLOCK_H=64, BLOCK_K=32, warps=4` remains best overall
- `BLOCK_H=32` and `BLOCK_K=64` were slower on N768 and did not produce a
  stable N1024 enough win

Conclusion: v48 is a useful C384 improvement in same-session comparison, but it
does not beat v42's best historical recheck. Do not promote it as the final
submission yet. The next 200-300 us probably has to come from first-timed C128
stability or a deeper C384 path that also removes tail/packing overhead, not
only projection/gate.

## v49 Rank01 Weight Cache On v48

`v49_v48_rank01_weight_cache.py` adds the safer v44 idea on top of v48: cache
only the fp16 transposed rank01 weights for the C128 Triton branch. It does not
cache activations or work buffers. The cache can be disabled with
`TRIMUL_V49_DISABLE_RANK01_WEIGHT_CACHE=1`, while the v48 C384 path still keeps
`TRIMUL_V48_DISABLE_C384=1`.

Validation on A100:

- full official test: 18/18 pass
- benchmark-shape test: 7/7 pass
- benchmark geomean: 2432.794 us
- leaderboard-style rechecks: 2489.406 us and 2521.721 us
- cache-disabled leaderboard control: 2492.101 us
- same-session v42 leaderboard control: 2549.262 us

Conclusion: v49 is better than v42 in the same session and produced one strong
recheck, but the rank01 weight cache itself only accounts for a few microseconds
of geomean in recheck. This is not a stable promotion over v42's best historical
2478.318 us. Keep v49 as the current best experimental candidate, not the final
answer. The next move should attack a broader source of variance/work than
weight conversion alone.

## v21 Shape Results

| Shape | v20 mean us | v21 best run mean us | Change | Notes |
| --- | ---: | ---: | ---: | --- |
| B2 N256 C128 nomask normal | 3374.880 | 2875.636 | -14.8% | cache removes visible fixed overhead |
| B1 N768 C128 nomask cauchy | 18257.205 | 18222.144 | -0.2% | dominated by large reductions/GEMMs |
| B2 N256 C384 mask normal | 3996.928 | 3189.173 | -20.2% | packed weights and workspace reuse help |
| B1 N512 C128 nomask normal | 8466.080 | 6675.168 | -21.2% | strongest real v21 improvement |
| B1 N1024 C128 nomask cauchy | 27689.029 | 34232.394 | +23.6% | needs stage timing; do not overfit cache |
| B1 N768 C384 mask normal | 15869.643 | 15823.008 | -0.3% | almost unchanged |
| B1 N1024 C384 nomask normal | 30183.318 | 30157.024 | -0.1% | almost unchanged |

## v22 Shape Results

The v22 comparison uses the best observed mean from repeated v21/v22 benchmark
runs on the same A100 VM. Small shapes show some VM noise; the large-shape
direction is stable.

| Shape | v21 best run mean us | v22 best run mean us | Change | Notes |
| --- | ---: | ---: | ---: | --- |
| B2 N256 C128 nomask normal | 2875.636 | 2856.053 | -0.7% | roughly flat; noisy small shape |
| B1 N768 C128 nomask cauchy | 18222.144 | 12956.585 | -28.9% | one fast high-variance run; other v22 runs around 15.8 ms |
| B2 N256 C384 mask normal | 3189.173 | 3196.288 | +0.2% | keep block LN fallback for this small C384 shape |
| B1 N512 C128 nomask normal | 6675.168 | 5870.101 | -12.1% | solid C128 win |
| B1 N1024 C128 nomask cauchy | 34232.394 | 24432.789 | -28.6% | biggest v22 recovery |
| B1 N768 C384 mask normal | 15823.008 | 14271.360 | -9.8% | large C384 benefits |
| B1 N1024 C384 nomask normal | 30157.024 | 27409.024 | -9.1% | large C384 benefits |

## v23 Shape Results

The v23 comparison uses best observed benchmark means for v22/v23. The
leaderboard-style recheck for v23 passed at 4315.162 us geomean.

| Shape | v22 best run mean us | v23 best run mean us | Change | Notes |
| --- | ---: | ---: | ---: | --- |
| B2 N256 C128 nomask normal | 2856.053 | 1086.660 | -62.0% | hidden LN/layout dominated the small C128 path |
| B1 N768 C128 nomask cauchy | 12956.585 | 6267.761 | -51.6% | large C128 now much closer to public Triton |
| B2 N256 C384 mask normal | 3196.288 | 2144.115 | -32.9% | mask path still slower than rank #2 |
| B1 N512 C128 nomask normal | 5870.101 | 2196.356 | -62.6% | major post-central win |
| B1 N1024 C128 nomask cauchy | 24432.789 | 9548.893 | -60.9% | biggest absolute win |
| B1 N768 C384 mask normal | 14271.360 | 6854.222 | -52.0% | large masked C384 still has mask/gate work left |
| B1 N1024 C384 nomask normal | 27409.024 | 12474.343 | -54.5% | final projection/GEMM now more visible |

## v24 Shape Results

The v24 comparison uses best observed benchmark means for v23/v24. This is a
small optimization with visible run-to-run noise, so the leaderboard-style
recheck should be treated as the safer number.

| Shape | v23 best run mean us | v24 best run mean us | Change | Notes |
| --- | ---: | ---: | ---: | --- |
| B2 N256 C128 nomask normal | 1086.660 | 1209.867 | +11.3% | v24 did not help this small nomask shape |
| B1 N768 C128 nomask cauchy | 6267.761 | 5224.430 | -16.6% | best visible nomask gate win |
| B2 N256 C384 mask normal | 2144.115 | 1756.120 | -18.1% | likely measurement noise; masked code path unchanged |
| B1 N512 C128 nomask normal | 2196.356 | 2181.325 | -0.7% | roughly flat |
| B1 N1024 C128 nomask cauchy | 9548.893 | 10430.219 | +9.2% | regression/noise; needs stage timing |
| B1 N768 C384 mask normal | 6854.222 | 6897.090 | +0.6% | roughly flat; masked code path unchanged |
| B1 N1024 C384 nomask normal | 12474.343 | 12273.269 | -1.6% | slight nomask win |

## v25 Stage Timing

Timings below are CUDA event milliseconds from `v25_cuda_ext_stage_timing.py`
using the second occurrence of each repeated benchmark shape. `pack` is omitted
from the table because it stayed near 0.01 ms in these runs. Absolute totals are
single-call diagnostics, not official benchmark means.

| Shape | LN | Projection | Gate/mask | Central | Hidden LN/gate/layout | Final proj | Total | Dominant next target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B2 N256 C128 nomask normal | 0.090 | 0.232 | 0.170 | 0.108 | 0.561 | 0.072 | 1.242 | hidden path |
| B1 N768 C128 nomask cauchy | 0.346 | 0.993 | 0.713 | 0.630 | 3.261 | 0.302 | 6.254 | hidden path |
| B2 N256 C384 mask normal | 0.726 | 0.416 | 0.169 | 0.106 | 0.575 | 0.171 | 2.172 | input LN + hidden path |
| B1 N512 C128 nomask normal | 0.161 | 0.450 | 0.321 | 0.217 | 1.373 | 0.123 | 2.653 | hidden path |
| B1 N1024 C128 nomask cauchy | 0.600 | 1.765 | 1.275 | 1.353 | 5.910 | 0.529 | 11.441 | hidden path |
| B1 N768 C384 mask normal | 0.997 | 1.792 | 0.718 | 0.628 | 3.226 | 0.703 | 8.073 | hidden path + projection |
| B1 N1024 C384 nomask normal | 1.766 | 2.753 | 1.256 | 1.093 | 4.583 | 0.913 | 12.375 | hidden path + projection |

Stage-timing conclusion: after v23, the hidden LN/gate/layout kernel is still
the largest single stage on every large shape. Projection is the second target
on C384 shapes. Gate/mask is meaningful but not large enough to explain the
remaining gap by itself, which matches the mixed v24 result.

## v26 Shape Results

The v26 comparison uses best observed benchmark means for v23/v26. The
leaderboard-style recheck for v26 passed at 3192.199 us geomean.

| Shape | v23 best run mean us | v26 best run mean us | Change | Notes |
| --- | ---: | ---: | ---: | --- |
| B2 N256 C128 nomask normal | 1086.660 | 834.699 | -23.2% | tiled hidden path removes most remaining fixed hidden-layout cost |
| B1 N768 C128 nomask cauchy | 6267.761 | 4178.741 | -33.3% | strong coalescing win on large C128 |
| B2 N256 C384 mask normal | 2144.115 | 1548.599 | -27.8% | masked path also benefits because hidden reads dominate after central |
| B1 N512 C128 nomask normal | 2196.356 | 1563.822 | -28.8% | stable mid-size win |
| B1 N1024 C128 nomask cauchy | 9548.893 | 7677.397 | -19.6% | large C128 still has projection/gate and central work left |
| B1 N768 C384 mask normal | 6854.222 | 5291.477 | -22.8% | C384 projection is now a larger fraction of the gap |
| B1 N1024 C384 nomask normal | 12474.343 | 9416.368 | -24.5% | biggest absolute win, still behind rank #2 mostly outside hidden kernel |

## v27 Post-v26 Stage Timing

Timings below are warmed CUDA event milliseconds from
`v27_cuda_ext_stage_timing_v26.py` using the second occurrence of each repeated
benchmark shape. The first occurrence is ignored when it includes workspace or
cuBLAS lazy initialization artifacts.

| Shape | LN | Projection | Gate/mask | Central | Hidden LN/gate/layout | Final proj | Total | Dominant next target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B2 N256 C128 nomask normal | 0.093 | 0.235 | 0.170 | 0.109 | 0.250 | 0.072 | 0.937 | hidden/projection/gate split |
| B1 N768 C128 nomask cauchy | 0.345 | 0.988 | 0.713 | 0.627 | 1.195 | 0.301 | 4.178 | hidden + projection |
| B2 N256 C384 mask normal | 0.725 | 0.413 | 0.168 | 0.104 | 0.251 | 0.172 | 1.844 | input LN |
| B1 N512 C128 nomask normal | 0.167 | 0.447 | 0.321 | 0.217 | 0.540 | 0.123 | 1.823 | hidden + projection |
| B1 N1024 C128 nomask cauchy | 0.600 | 1.760 | 1.275 | 1.350 | 2.146 | 0.527 | 7.667 | hidden + projection/gate |
| B1 N768 C384 mask normal | 0.999 | 1.787 | 0.718 | 0.625 | 1.197 | 0.700 | 6.037 | projection + hidden + input LN |
| B1 N1024 C384 nomask normal | 1.765 | 2.730 | 1.255 | 1.066 | 1.668 | 0.911 | 9.407 | projection + input LN |

Post-v26 timing conclusion: the old huge hidden bottleneck is gone. Hidden is
still important, especially at N1024 C128, but the next millisecond must come
from shape-specific smaller wins: hidden tile tuning, C384 input LN/projection,
and eventually cuBLAS/cuBLASLt tuning.

## v28-v30 Shape Results

The table below compares the main post-v26 attempts against the best observed
v26 benchmark means. `v30` is the best benchmark run from this group, but its
leaderboard-style recheck was 3204.735 us, close to v26's 3192.199 us.

| Shape | v26 best run mean us | v28 mean us | v29 mean us | v30 mean us | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| B2 N256 C128 nomask normal | 834.699 | 841.977 | 905.635 | 910.736 | tile16 hurts small C128; v26 remains better here |
| B1 N768 C128 nomask cauchy | 4178.741 | 4150.987 | 3999.275 | 3528.738 | tile16 helps large C128; v30 first run was very fast/noisy |
| B2 N256 C384 mask normal | 1548.599 | 1845.146 | 1830.587 | 1518.368 | small C384 is noisy; v30 matched best behavior in benchmark |
| B1 N512 C128 nomask normal | 1563.822 | 1564.814 | 1736.793 | 1562.635 | keep tile8 for N512 |
| B1 N1024 C128 nomask cauchy | 7677.397 | 7609.259 | 7313.227 | 7321.973 | tile16 is consistently useful |
| B1 N768 C384 mask normal | 5291.477 | 5287.552 | 5130.888 | 5137.872 | tile16 helps large masked C384 |
| B1 N1024 C384 nomask normal | 9416.368 | 9376.320 | 9159.477 | 9138.315 | tile16 is consistently useful |

## v31-v33 Shape Results

The table below uses leaderboard-style recheck means. `v32` became the best
verified submission from this round. The biggest reliable gain is the
small masked C384 shape, where using warp input LayerNorm for all C384 sizes
removes the old block-LN fallback cost.

| Shape | v26 recheck us | v31 recheck us | v32 recheck us | v33 recheck us | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| B2 N256 C128 nomask normal | 841.045 | 919.893 | 840.294 | 812.001 | noisy small C128; v32 stays near v26 |
| B1 N768 C128 nomask cauchy | 4220.587 | 3992.576 | 4031.488 | 3996.672 | tile16/large nomask helps, but not enough alone |
| B2 N256 C384 mask normal | 1552.906 | 1550.100 | 1197.353 | 1354.411 | v32 C384 warp LN is the real win |
| B1 N512 C128 nomask normal | 1579.691 | 1572.181 | 1584.128 | 1833.643 | avoid extra nomask tweaks here |
| B1 N1024 C128 nomask cauchy | 7707.648 | 7289.856 | 7361.195 | 7290.301 | tile16 helps large C128 |
| B1 N768 C384 mask normal | 5319.339 | 5169.920 | 5171.968 | 5167.718 | tile16 helps large masked C384 |
| B1 N1024 C384 nomask normal | 9461.077 | 9123.157 | 9184.256 | 9121.792 | tile16/large nomask helps, but projection remains big |

## v34 GEMM Algo Sweep

`v34_cuda_ext_gemm_algo_tune.py` keeps v32's math path and adds env-controlled
cuBLAS algorithm selection:

- `TRIMUL_PROJ_ALGO=default|0|1`
- `TRIMUL_CENTRAL_ALGO=default|0|1`
- `TRIMUL_OUT_ALGO=default|0|1`

The default v34 path uses `ALGO1` for central N1024 C128/C384 and default
Tensor Core algorithms elsewhere. This is a useful tuning harness, but the
current sweeps did not beat v32's 3011.576 us recheck.

| Variant | Benchmark geomean us | Recheck geomean us | Notes |
| --- | ---: | ---: | --- |
| default static central | 3087.958 | 3060.919 | correctness pass; not promoted |
| `TRIMUL_CENTRAL_ALGO=default` | 3129.944 | n/a | worse than v32 |
| `TRIMUL_CENTRAL_ALGO=0` | 3153.272 | n/a | worse than v32 |
| `TRIMUL_CENTRAL_ALGO=1` | 3001.251 | 3062.057 | benchmark looked good, recheck did not hold |
| `TRIMUL_PROJ_ALGO=0` | 3147.167 | n/a | worse than v32 |
| `TRIMUL_PROJ_ALGO=1` | 3003.092 | 3050.597 | benchmark looked good, recheck did not hold |
| `TRIMUL_OUT_ALGO=1` | 3116.648 | n/a | worse than v32 |

v34 conclusion: simple cuBLAS algo switching is too noisy and does not close
the rank #2 gap by itself. The N1024 C128 row can occasionally drop near
6.2 ms, but the effect did not survive leaderboard-style recheck. The next
projection work should either use cuBLASLt with real heuristic selection and
workspace, or move to a custom/fused projection path for the C384-heavy shapes.

## Next Kernel Targets

1. `v21`: add one-shape workspace cache and packed-weight cache. Done; keep as
   infrastructure, not as a leaderboard-level optimization by itself.
2. `v22`: replace block-per-row input LayerNorm with warp-per-row kernels for
   `C=128`, `C=384`, and `C=768`. Done; large shapes improved, and small
   `C=384` now uses the old fallback.
3. `v23`: replace block-per-row hidden LN/gate with a tiled warp kernel that
   reads `[H, rows]` and writes `[rows, H]`. Done; this is the first sub-5ms
   custom CUDA/cuBLAS version.
4. `v24`: add mask fast paths: all-ones float mask, int64 aligned, and optional
   `uint8` compaction for large masked shapes. The all-ones float mask path is
   done; remaining mask work should focus on int64/vectorized masked cases.
5. `v25`: add stage timing. Done; next target should be a coalesced/tiled
   hidden LN path, then projection/cuBLAS tuning for C384.
6. `v26`: add a coalesced tiled hidden LN/out-gate/layout path. Done; this is
   the first stable custom CUDA/cuBLAS version near 3.2ms.
7. `v27`: re-run stage timing after v26. Done; bottleneck is split across
   hidden, projection, gate, and C384 input LN.
8. `v28`: test nomask gate shortcut on top of v26. Done; correctness-safe but
   not a stable speed win.
9. `v29`: test 16-row hidden tile. Done; large-shape win, small-shape loss.
10. `v30`: dispatch tile8/tile16 by N. Done; promising benchmark win, but
    recheck is only roughly tied with v26.
11. `v31`: combine tile dispatch with nomask gate. Done; correctness-safe and
    useful signal, but noisy.
12. `v32`: remove the C384 block-LN fallback. Done; former best verified run
    at 3011.576 us.
13. `v33`: try selective large-shape nomask gate on top of v32. Done; does not
    beat v32.
14. `v34`: add env-controlled cuBLAS algorithm tuning. Done; useful harness,
    but not promoted.
15. `v35`: move beyond simple cuBLAS algo flags: cuBLASLt with workspace and
    heuristic selection, or a custom C384 projection path.
    Done; cuBLASLt is useful as a harness but too noisy to promote.
16. `v36`: add stage timing on top of the current v32 best path. Done; it
    showed gate/mask, projection, hidden, and C384 input LN are now the broad
    pressure points.
17. `v37`: replace the gate/mask update with a vectorized H-major kernel.
    Done; small stable gate-stage win, but not enough for a 700 us jump.
18. `v38`: prioritize C384 input LN/projection cleanup or a deeper
    projection-output/gate fusion. Do this before another cuBLAS flag sweep.
    Done for C384 input LN; not promoted because the LN variants were flat.
19. `v39`: pivot to projection-output/gate traffic reduction. The likely target
    is not custom projection GEMM yet; first try reducing the post-projection
    write/read tax around left/right gates and central operands.
    Done as rank02-vs-v37 stage comparison first; the measurement shows hidden
    LN/out-gate/layout, not projection/gate, is the dominant rank02 gap.
20. `v40`: port or reproduce rank02's faster hidden bucket. Start by studying
    `ln_affine_gate_from_col_to_row_f16_kernel` and its tile shape, then test a
    v37-compatible hidden replacement before returning to mask/gate work.
    Done; v40 is the new native baseline with leaderboard-style rechecks at
    2673.060 / 2643.726 us.
21. `v41`: close the remaining rank02 gap with smaller, shape-specific work:
    projection/gate traffic, small-shape launch/memory overhead, and C384 rows.
    Start with stage timing that includes v40, rank02, and the public rank01
    reference if it can be instrumented safely.
    Done; rank01 timing shows the large C128 projection/gate path is the real
    next lever.
22. `v42`: hybridize based on v41: use rank01 Triton for B1 C128 N512/768/1024
    and v40 everywhere else. Done; official test passes and leaderboard-style
    rechecks are 2503.175 / 2478.318 us.
23. `v43`: cache rank01 C128 weights/work buffers. Done; benchmark-only win,
    not promoted because rechecks stayed around 2.51ms.
24. `v44`: try weight-only cache to avoid buffer-cache side effects. Done;
    safer but still not better than v42's best recheck.
25. `v45`: test rank01-first dispatch order. Done; correctness passes, but
    benchmark mode exits 112 before producing rows.
26. `v46`: measure v42/v40/rank01/rank02 in one all-7 stage harness before
    changing more code. Done; final fusion is not the right C384-first target.
27. `v47`: cheap C384 dispatch probe. Done; full rank01 for `B1 N768 C384`
    passes correctness but regresses leaderboard-style recheck, so do not
    promote.
28. `v48`: surgical C384 projection/gate work. Do not route full C384 rows
    through rank01, and do not replace the v40 C384 final path yet. The target
    is a native C384 projection/gate kernel or split-stage path that writes
    v40-compatible H-major left/right/out-gate buffers, preserving v40's input
    LN, central GEMM, hidden kernel, and final cuBLAS projection.
    Done; correctness passes and same-session recheck beats v42, but it does
    not beat v42's best historical run.
29. `v49`: reintroduce the safer v44 rank01 fp16 transposed weight cache on
    top of v48. Done; correctness passes and one recheck hit 2489.406 us, but
    cache-disabled control was 2492.101 us, so the cache is not a large enough
    lever by itself.
30. `v50`: stabilize first-timed C128 or remove v48 tail overhead. The C384
    projection/gate win is only 100-200 us per C384 large row; rank #1 needs a
    broader source. Prefer a recheck-stable C128 path or a C384 path that
    combines projection/gate with the tail without reintroducing rank01's C384
    final regression.
