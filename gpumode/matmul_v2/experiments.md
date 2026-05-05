# matmul_v2 Experiments

기준 리더보드: `matmul_v2`, GPU: `L4`

## Environment

| Field | Value |
| --- | --- |
| GCP project | `nemo-488500` |
| Preferred instance | `cuda-l4-dev-lesson10` |
| Zone | `us-west1-b` |
| Machine | `g2-standard-4` |
| GPU | `1x nvidia-l4` |
| Provisioning | `SPOT` |

## Submission Log

| Version | File | Mode | Score us | Rank | Correct | Notes |
| --- | --- | --- | ---: | ---: | --- | --- |
| v0_safe | `submissions/v0_safe.py` | test | N/A | N/A | pass | GCP L4 official harness clone |
| v0_safe | `submissions/v0_safe.py` | benchmark | 2992.352 | N/A | pass | GCP L4 official harness clone; mean sum |
| v0_safe | `submissions/v0_safe.py` | leaderboard | 3010.897 | N/A | pass | Local ranked-mode proxy |
| v0_safe | `submissions/v0_safe.py` | ranked | TBD | TBD | TBD | Submit only after benchmark sanity |
| v1_hybrid_large | `submissions/v1_hybrid_large.py` | benchmark | fail | N/A | fail | `2048^3` mismatch, do not submit |
| v1_hybrid_large_keep2 | `submissions/v1_hybrid_large_keep2.py` | benchmark | 2938.236 | N/A | pass | Custom only `2048x3072`, `4096x5120` |
| v1_hybrid_large_keep2 | `submissions/v1_hybrid_large_keep2.py` | leaderboard | 3036.015 | N/A | pass | Ranked proxy worse than v0 due final-shape variance |
| v1_hybrid_2048x3072_only | `submissions/v1_hybrid_2048x3072_only.py` | leaderboard | 3012.943 | N/A | pass | No clear ranked-mode win over v0 |
| v2_bigshape_bk32 | `submissions/v2_bigshape_bk32.py` | benchmark | 2882.787 | N/A | pass | `128x128x32`, custom on two big shapes |
| v2_bigshape_bk32 | `submissions/v2_bigshape_bk32.py` | leaderboard | 2904.946 | N/A | pass | First stable real candidate |
| v2_bigshape_bk32 | `submissions/v2_bigshape_bk32.py` | leaderboard | 2909.358 | N/A | pass | Repeat |
| v3_bigshape_bk32_grouped | `submissions/v3_bigshape_bk32_grouped.py` | benchmark | 2827.212 | N/A | pass | Shape-specific `GROUP_M`: 8/16 |
| v3_bigshape_bk32_grouped | `submissions/v3_bigshape_bk32_grouped.py` | leaderboard | 2862.659 | N/A | pass | Best candidate so far |
| v3_bigshape_bk32_grouped | `submissions/v3_bigshape_bk32_grouped.py` | leaderboard | 2856.437 | N/A | pass | Repeat, stable |
| v3_bigshape_bk32_grouped | `submissions/v3_bigshape_bk32_grouped.py` | official benchmark | N/A | N/A | pass | Submission `780609`; Modal L4 largest shape `2.08 ms` |
| v4_bigshape_cache_hint | `submissions/v4_bigshape_cache_hint.py` | benchmark | 2806.145 | N/A | pass | `.cg`/eviction hints on largest shape |
| v4_bigshape_cache_hint | `submissions/v4_bigshape_cache_hint.py` | leaderboard | 2840.097 | N/A | pass | GCP proxy |
| v4_bigshape_cache_hint | `submissions/v4_bigshape_cache_hint.py` | official benchmark | N/A | N/A | pass | Submission `780610`; Modal L4 largest shape `1980 ± 1.6 us` |
| v4_bigshape_cache_hint | `submissions/v4_bigshape_cache_hint.py` | official ranked | 2076.331 | 1 | pass | Submission `780611`; became L4 rank 1 on 2026-05-05 |

## Per-Shape Log

| Version | Shape `(M,N,K)` | Mean us | Best us | Std us | Runs | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| v0_safe | `(128,128,128)` | 8.192 | 8.192 | 0.000 | 3 |  |
| v0_safe | `(256,256,256)` | 12.002 | 10.240 | 0.987 | 100 |  |
| v0_safe | `(512,512,512)` | 14.787 | 13.312 | 1.082 | 100 |  |
| v0_safe | `(1024,1024,1024)` | 64.551 | 60.416 | 1.634 | 100 |  |
| v0_safe | `(2048,2048,2048)` | 263.168 | 263.168 | 0.000 | 3 |  |
| v0_safe | `(1024,1536,1024)` | 54.292 | 52.224 | 0.987 | 100 |  |
| v0_safe | `(2048,3072,2048)` | 356.011 | 355.328 | 0.591 | 3 |  |
| v0_safe | `(4096,5120,4096)` | 2219.349 | 2217.984 | 1.182 | 3 | Dominates baseline |

## Commands

Discord candidate:

```text
/leaderboard submit test gpu:L4 leaderboard_name:matmul_v2 script:v3_bigshape_bk32_grouped.py
/leaderboard submit benchmark gpu:L4 leaderboard_name:matmul_v2 script:v3_bigshape_bk32_grouped.py
/leaderboard submit ranked gpu:L4 leaderboard_name:matmul_v2 script:v3_bigshape_bk32_grouped.py
```

Popcorn CLI candidate:

```bash
popcorn-cli submit --gpu L4 --leaderboard matmul_v2 --mode test submissions/v3_bigshape_bk32_grouped.py
popcorn-cli submit --gpu L4 --leaderboard matmul_v2 --mode benchmark submissions/v3_bigshape_bk32_grouped.py
popcorn-cli submit --gpu L4 --leaderboard matmul_v2 --mode leaderboard submissions/v3_bigshape_bk32_grouped.py
```

GCP local proxy:

```bash
./scripts/gcp_eval_submission.sh test submissions/v0_safe.py
./scripts/gcp_eval_submission.sh benchmark submissions/v0_safe.py
./scripts/gcp_eval_submission.sh leaderboard submissions/v0_safe.py
python3 scripts/sweep_large_triton.py --iters 5 --warmup 1
```

## Sweep Summary

`scripts/sweep_large_triton.py` on `cuda-l4-dev-lesson10`, seed `123456`, 2026-05-05:

| Shape | Best correct config | Mean us | Correctness notes |
| --- | --- | ---: | --- |
| `(2048,2048,2048)` | none | N/A | All tested Triton configs had max diff `0.5`; keep torch |
| `(2048,3072,2048)` | `128x128x64_w4_s4` | 340.992 | Correct, faster than torch |
| `(4096,5120,4096)` | `128x128x64_w4_s4` | 2186.240 | Correct and fast in fixed-seed sweep, noisy in leaderboard proxy |
| `(1024,1024,1024)` | none | N/A | All tested Triton configs had max diff `0.25`; keep torch |
| `(1024,1536,1024)` | `128x128x64_w8_s4` | 60.826 | Correct but slower than torch baseline; keep torch |

Decision superseded by v3: do not ranked-submit `v0_safe` or `v1_hybrid_large_keep2`; keep them as baselines only.

## v2/v3 Sweep Summary

`scripts/sweep_bigshape_v2.py` found `128x128x32_w4_s4_g8` as the first robust improvement:

| Shape | Config | Mean us | Delta vs torch in sweep |
| --- | --- | ---: | ---: |
| `(4096,5120,4096)` | `128x128x32_w4_s4_g8` | 2131.968 | -131.891 |
| `(2048,3072,2048)` | `128x128x32_w4_s4_g8` | 326.246 | -45.261 |

`scripts/sweep_bk32_focused.py` then showed `GROUP_M=16` is better for the largest shape, while `GROUP_M=8` remains better for `(2048,3072,2048)`:

| Shape | Best config | Mean us | Notes |
| --- | --- | ---: | --- |
| `(4096,5120,4096)` | `128x128x32_w4_s4_g16` | 2082.304 | Best focused sweep result |
| `(2048,3072,2048)` | `128x128x32_w4_s4_g8` | 327.936 | Best stable result |

Current decision: `v3_bigshape_bk32_grouped.py` is the first file worth official GPUMODE benchmark submission. It is still not near the current L4 top score, but it is clearly better than `torch.mm(out=c)` on the GCP L4 proxy.

## v4 Official Result

`v4_bigshape_cache_hint.py` adds cache hints only for `(4096,5120,4096)`:

- `tl.load(A, cache_modifier=".cg", eviction_policy="evict_first")`
- `tl.load(B, cache_modifier=".cg", eviction_policy="evict_last")`
- keeps `(2048,3072,2048)` on the v3 default Triton kernel
- keeps all other shapes on `torch.mm(out=c)`

Official GPUMODE L4 result, 2026-05-05:

| Rank | Score us | User | File | Submission |
| ---: | ---: | --- | --- | ---: |
| 1 | 2076.331 | brianyu | `v4_bigshape_cache_hint.py` | 780611 |
| 2 | 2200.917 | iharryli | `ori_submission.py` | 512469 |
| 3 | 2226.859 | mreso | `solution.py` | 512815 |

Current decision: `v4_bigshape_cache_hint.py` is the active ranked submission. The next optimization target is no longer "get onto the board"; it is defending/improving the `~2.076 ms` score.

## A100 Migration

GCP A100 VM:

| Field | Value |
| --- | --- |
| Instance | `cuda-a100-dev-matmul-v2` |
| Zone | `us-central1-a` |
| Machine | `a2-highgpu-1g` |
| Provisioning | `SPOT` |
| GPU | `NVIDIA A100-SXM4-40GB` |
| Driver | `580.126.20` |
| Python stack | `torch 2.11.0+cu130`, `triton 3.6.0` |

Created on 2026-05-05. The image did not include `pip`, `torch`, or `triton` by default, so `python3-pip`, build tools, PyTorch, and Triton were installed into the user environment. The VM was stopped after the first measurements.

Initial A100 result:

| Version | Mode | Proxy score us | Correct | Notes |
| --- | --- | ---: | --- | --- |
| `a100_v0_safe.py` | test | N/A | pass | `torch.mm(out=c)` |
| `a100_v0_safe.py` | benchmark | 1000.542 | pass | A100 baseline |
| `a100_v1_l4_v4_port.py` | test | N/A | pass | L4 v4 port |
| `a100_v1_l4_v4_port.py` | benchmark | 1275.986 | pass | Much slower than torch baseline |

Large-shape quick sweep on A100:

| Shape | Torch mean us | Best tested Triton mean us | Winner |
| --- | ---: | ---: | --- |
| `(4096,5120,4096)` | 788.890 | 876.339 | Torch |
| `(2048,3072,2048)` | 144.384 | 159.130 | Torch |

Decision: the L4-winning Triton strategy does not migrate directly to A100. On this A100 stack, cuBLAS via `torch.mm(out=c)` is already stronger than the tested Triton tiles for the two dominant shapes. Next A100 work should either submit `a100_v0_safe.py` as the official A100 baseline or pivot to cublasLt/CUTLASS rather than continuing the same Triton tile family.

## A100 Official Result

Official GPUMODE A100 submissions, 2026-05-05:

| Version | Mode | Submission | Score us | Rank | Correct | Notes |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `a100_v0_safe.py` | official test | 780612 | N/A | N/A | pass | Modal `A100-SXM4-80GB`, public tests passed |
| `a100_v0_safe.py` | official benchmark | 780613 | 747 | N/A | pass | Modal `A100-SXM4-80GB`, benchmark output only |
| `a100_v0_safe.py` | official ranked | 780614 | 634.197 | 2 | pass | Modal `A100 80GB PCIe`, ranked benchmark |

Current A100 leaderboard snapshot after submission `780614`:

| Rank | Score us | User | File | Submission |
| ---: | ---: | --- | --- | ---: |
| 1 | 629.760 | rajesh0042 | `matmul_v6.py` | 545267 |
| 2 | 634.197 | brianyu | `a100_v0_safe.py` | 780614 |
| 3 | 635.221 | Kernel-Zhang | `ref.py` | 780441 |
| 4 | 641.536 | Elan Zainos Corona | `tuplas3.4.py` | 674172 |
| 5 | 641.707 | dannywillowliu-uchi | `submission.py` | 614108 |

Gap to A100 rank 1 is about `4.437 us`, or `0.70%`. That is small enough that the next step should be a very narrow A100 rank-1 attack, not a broad Triton sweep. The highest-leverage target is shaving Python/PyTorch dispatch overhead or forcing a better cuBLAS/cuBLASLt path for the final benchmark shape.

Thin A100 API variants created after rank 2:

| File | Idea | Official test | Official benchmark |
| --- | --- | --- | --- |
| `submissions/a100/a100_v1_return_mm.py` | `return torch.mm(a, b, out=c)` | pass | pending |
| `submissions/a100/a100_v1_matmul_out.py` | `return torch.matmul(a, b, out=c)` | pass | pending |
| `submissions/a100/a100_v1_addmm_out.py` | `return torch.addmm(c, a, b, beta=0, out=c)` | pass | pending |
| `submissions/a100/a100_v1_aten_mm_out.py` | direct `torch.ops.aten.mm.out` | pass | pending |

Benchmarking these variants is blocked by the GPUMODE hourly submission limit (`6/6` submissions used). Resume with benchmark mode after the cooldown instead of spending more submissions on new ideas.

## A100 cuBLAS Path Analysis

GCP A100 Spot VM was restarted on 2026-05-05 for local profiling, then stopped again after logs were copied back. Local profiling files:

- `logs/a100/profile/nsys_mm_stats.txt`
- `logs/a100/profile/nsys_aten_mm_stats.txt`
- `logs/a100/profile/ncu_aten_mm_sudo.txt`

Isolated final-shape timing on GCP `A100-SXM4-40GB`, shape `(4096,5120,4096)`:

| Variant | Mean us | Median us | Best us | Iters | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `torch.mm(a, b, out=c)` | 799.621 | 797.696 | 794.624 | 50 | Baseline API path |
| `torch.ops.aten.mm.out(a, b, out=c)` | 687.596 | 687.104 | 683.008 | 50 | Same op family, much thinner local timing |

Nsight Systems shows both variants launch the same cuBLAS kernel:

```text
ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_64x3_nn
```

So the local `aten_mm` win is not from a different visible GEMM kernel name. It is likely from a thinner PyTorch/ATen call path, less stream idle time inside the CUDA-event interval, or a small cuBLAS state/heuristic difference not reflected in the demangled kernel name.

Nsight Compute on `aten_mm` captured:

| Metric | Value |
| --- | ---: |
| Kernel duration | `777.95 us` under NCU overhead |
| Compute throughput | `91.13%` |
| Memory throughput | `48.57%` |
| DRAM throughput | `27.04%` |
| L2 hit rate | `78.64%` |

Interpretation: the cuBLAS kernel is strongly compute-bound and already highly optimized. Beating it with a plain Triton GEMM is unlikely. The realistic A100 rank-1 path is still to use the best cuBLAS/cuBLASLt/CUTLASS path with the thinnest possible submission overhead. The next official benchmark priority is `a100_v1_aten_mm_out.py`.
