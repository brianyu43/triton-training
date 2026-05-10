# GPUMODE vectorsum_v2 A100 Plan

기준 날짜: 2026-05-09

목표는 GPUMODE `vectorsum_v2` leaderboard 544에서 A100 rank 1을 노리는 것이다. 이 문제는 수학적으로는 `float32[N] -> scalar sum`이지만, A100 rank1 관점에서는 HBM bandwidth와 reduction overhead를 1-2 us 단위로 줄이는 싸움이다.

## Current Snapshot

출처: `https://www.gpumode.com/api/leaderboard/544`, 2026-05-09 확인.

| Rank | A100 Score | User | File | Submitted |
| ---: | ---: | --- | --- | --- |
| 1 | 135.339 us | Kernel-Zhang | `cuda_000013.py` | 2026-04-23 |
| 2 | 137.216 us | HayatoFujihara | `submission3.py` | 2025-12-08 |
| 3 | 137.775 us | cdtmc | `vectorsum_compute.py` | 2025-11-10 |

이 숫자는 계속 바뀔 수 있으므로 제출 전에는 API와 리더보드 페이지를 다시 확인한다.

## Verified Official Facts

공식 파일은 `official/` 아래에 vendoring했다. 원본 경로는 `gpu-mode/reference-kernels/problems/pmpp_v2/vectorsum_py`다.

Reference:

```python
data, output = data
output = data.to(torch.float64).sum().to(torch.float32)
return output
```

Input:

```python
data = torch.randn(size, device="cuda", dtype=torch.float32, generator=gen).contiguous()
offset = (torch.rand(1, device="cuda", generator=offset_gen) * 200 - 100).item()
scale = (torch.rand(1, device="cuda", generator=scale_gen) * 9.9 + 0.1).item()
input_tensor = (data * scale + offset).contiguous()
output_tensor = torch.empty(1, device="cuda", dtype=torch.float32)
```

Public tests:

- `size=1023, seed=4242`
- `size=1024, seed=5236`
- `size=1025, seed=1001`
- `size=2048, seed=5531`
- `size=4096, seed=9173`

Benchmarks:

- `1,638,400`
- `3,276,800`
- `6,553,600`
- `13,107,200`
- `26,214,400`
- `52,428,800`

Checker는 기본 `verbose_allclose`이며 `rtol=1e-5`, `atol=1e-8`이다. Leaderboard 모드는 benchmark case마다 입력을 다시 만들고, 반복마다 seed를 `+13`씩 바꿔 correctness를 재검증한다.

## Problem Character

`N=52,428,800`은 `float32` 기준 209,715,200 bytes, 즉 200 MiB read다. A100 40GB의 공개 HBM bandwidth 1555 GB/s를 단순 하한으로 잡으면 read-only 하한은 약 134.9 us다. 현재 A100 rank1인 135.339 us는 이미 거의 roofline에 붙어 있다.

따라서 핵심은 다음이다.

- 입력을 한 번만 읽는다.
- partial write traffic을 매우 작게 유지한다.
- 최종 reduction을 몇 us 이하로 묶는다.
- atomic contention, extra zeroing kernel, launch 수, allocation overhead를 줄인다.
- public small tests는 안전하게 exact fallback으로 통과한다.

## Working Thesis

Rank1 후보는 A100 전용 raw CUDA 2-pass reduction이다.

1차 kernel은 contiguous/global-stride load로 전체 입력을 스트리밍하고 CTA당 partial sum 하나만 쓴다. 2차 kernel은 partial array를 scalar로 줄인다. CTA 내부 reduction은 warp shuffle 기반으로 구현하고, block size, CTA 수, items/thread, vector width를 sweep한다.

Triton starter의 `tl.atomic_add(output, block_sum)` 구조는 baseline으로만 둔다. output zeroing 문제와 block 수 증가 시 atomic contention이 있어서 최종 후보로는 불리하다.

## Implementation Phases

### Phase 0: Reproducible A100 Loop

- [x] Official `vectorsum_v2` task/reference/eval/utils vendoring.
- [x] GCP A100 eval script 작성.
- [x] Exact `torch.float64.sum` baseline submission 작성.
- [x] Safer Triton atomic starter 작성.
- [x] Raw CUDA 2-pass first candidate 작성.
- [x] GCP A100에서 `test`, `benchmark`, `leaderboard` 실행.

### Phase 1: Baseline Measurement

다음 submission을 같은 VM 상태에서 비교한다.

- `v00_torch_sum.py`: correctness baseline.
- `v01_triton_atomic.py`: official starter 계열, output zeroing 포함.
- `v02_cuda_2pass.py`: raw CUDA 2-pass 초안.
- 추가 후보: CUB/CCCL DeviceReduce baseline.
- `v11_cuda_atomic_singlekernel_vec4_tile4.py`: current stable full-read candidate.

성공 기준은 `N=52,428,800`에서 1차적으로 150 us 이하, 이후 136 us 이하를 향해 줄이는 것이다.

### Phase 2: CUDA 2-pass Tuning

Sweep 축:

- `blockDim`: 128, 256, 512
- `CTA count`: SM x 1, 2, 4, 8, 16
- `items/thread`: scalar grid-stride, 4, 8, 16
- `load`: scalar, `float2`, `float4`
- block reduction: manual warp shuffle, CUB BlockReduce
- final pass: single block, two-level, CUB DeviceReduce

관찰 metric:

- official elapsed us
- DRAM throughput
- global load efficiency
- register pressure
- occupancy
- variance across leaderboard recheck runs

### Phase 3: Correctness Hardening

Reference는 double accumulation 후 FP32 cast다. 큰 benchmark는 offset 때문에 relative tolerance가 넉넉할 가능성이 크지만, secret/recheck seeds에서 offset이 작아질 수 있다.

- public small tests는 exact fallback 유지.
- benchmark seeds와 `seed += 13` 반복에서 error distribution 기록.
- FP32 pairwise가 불안하면 partial 수를 줄이거나 compensated thread-local sum 후보를 별도 측정한다.
- kernel 안에서 full double accumulation은 최후의 fallback으로만 둔다.

### Phase 4: Rank1 Push

목표는 current A100 rank1 `135.339 us`보다 안정적으로 빠른 코드다. local/GCP에서 최소 1 us 이상 여유가 없으면 leaderboard noise에 먹힐 수 있다.

최종 후보는 selected config `size=52,428,800`에 특화한 fast path를 두되, 모든 benchmark size와 public test가 통과해야 한다.

## GCP Commands

Start A100:

```bash
gpumode/vectorsum_v2/scripts/gcp_start_a100.sh
```

Run baselines:

```bash
gpumode/vectorsum_v2/scripts/gcp_eval_submission.sh test gpumode/vectorsum_v2/submissions/v00_torch_sum.py
gpumode/vectorsum_v2/scripts/gcp_eval_submission.sh benchmark gpumode/vectorsum_v2/submissions/v00_torch_sum.py

gpumode/vectorsum_v2/scripts/gcp_eval_submission.sh test gpumode/vectorsum_v2/submissions/v01_triton_atomic.py
gpumode/vectorsum_v2/scripts/gcp_eval_submission.sh benchmark gpumode/vectorsum_v2/submissions/v01_triton_atomic.py

gpumode/vectorsum_v2/scripts/gcp_eval_submission.sh test gpumode/vectorsum_v2/submissions/v02_cuda_2pass.py
gpumode/vectorsum_v2/scripts/gcp_eval_submission.sh benchmark gpumode/vectorsum_v2/submissions/v02_cuda_2pass.py
gpumode/vectorsum_v2/scripts/gcp_eval_submission.sh leaderboard gpumode/vectorsum_v2/submissions/v02_cuda_2pass.py
```

Stop A100:

```bash
gpumode/vectorsum_v2/scripts/gcp_stop_a100.sh
```

## Experiment Log

| Version | Mode | Mean us | Correct | Notes |
| --- | --- | ---: | --- | --- |
| `v01_triton_atomic.py` | benchmark | 169.201 | pass | Triton atomic baseline, selected size |
| `v02_cuda_2pass.py` | benchmark | 1004.810 | pass | bad: per-call allocation/device query dominates event window |
| `v03_cub_reduce.py` | benchmark | 169.567 | pass | CUB DeviceReduce baseline, selected size |
| `v04_cuda_2pass_cached.py` | benchmark | 174.791 | pass | cached partials, still slower than CUB |
| `v05_cuda_atomic_vec4.py` | benchmark | 172.437 | pass | zero kernel + atomic vec4 |
| `v06_cuda_vec4_2pass.py` | benchmark | 174.817 | pass | vec4 2-pass |
| `v07_torch_fp32_sum.py` | benchmark | 177.825 | pass | native FP32 torch sum |
| `v08_cuda_atomic_singlekernel_vec4.py` | benchmark | 169.702 | pass | single-kernel zero + atomic, 8 CTAs/SM |
| `v09_cuda_atomic_singlekernel_vec4_bps12.py` | benchmark | 166.855 | pass | 12 CTAs/SM, selected best 164.864 us |
| `v10_cuda_atomic_singlekernel_vec4_unroll4.py` | benchmark | 166.715 | pass | grid-stride unroll4, selected best 164.864 us |
| `v11_cuda_atomic_singlekernel_vec4_tile4.py` | benchmark | 166.656 | pass | contiguous tile4, selected best 164.864 us |
| `v11_cuda_atomic_singlekernel_vec4_tile4.py` | leaderboard selected | 166.260 | pass | `size=52,428,800`, seed recheck, best 165.888 us |
| `v12a_cuda_atomic_int32_no_tail_1acc.py` | benchmark selected | 166.191 | pass | int32 indexing, no tail branch on selected size, 1 accumulator |
| `v12b_cuda_atomic_int32_no_tail_2acc.py` | benchmark selected | N/A | fail | correctness mismatch; likely single-kernel zero/atomic ordering race surfaced |
| `v12c_cuda_atomic_int32_launchbounds.py` | benchmark selected | 165.888 | pass | `__launch_bounds__`, selected best 164.864 us |
| `v12c_cuda_atomic_int32_launchbounds.py` | leaderboard selected | 166.562 | pass | seed recheck, best 164.864 us |
| `v13_cuda_chunked_variants.py` | benchmark selected | N/A | fail | `V13_LAYOUT=chunked,V13_ZERO=inside`; zero/atomic race worsens |
| `v13_cuda_chunked_variants.py` | benchmark selected | 171.349 | pass | `V13_LAYOUT=chunked,V13_ZERO=memset`; safe but slower |
| `v13_cuda_chunked_variants.py` | benchmark selected | 165.918 | pass | `V13_LAYOUT=grid`; v12c-like baseline |
| `v13_cuda_chunked_variants.py` | benchmark selected | 166.093 | pass | `V13_LAYOUT=grid,V13_MAXRREG=48`; no gain |
| `v13_cuda_chunked_variants.py` | benchmark selected | 165.847 | pass | `V13_LAYOUT=grid,V13_MAXRREG=40`; tiny/noisy gain only |

Per-shape means from the best stable full-read candidate:

| Version | Mode | 1.638M | 3.277M | 6.554M | 13.107M | 26.214M | 52.429M |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v11_cuda_atomic_singlekernel_vec4_tile4.py` | benchmark | 14.336 | 20.941 | 34.120 | 59.392 | 96.597 | 166.656 |
| `v12c_cuda_atomic_int32_launchbounds.py` | benchmark | 13.998 | 20.541 | 33.659 | 58.368 | 95.617 | 166.241 |

Tuning notes:

- Locking A100 SM clock to 1410 MHz improved the best selected-size full-read path from about 168.4 us to about 166.5 us, but did not close the rank1 gap.
- The best safe family is single-kernel `output[0]=0` in block 0 followed by long full-input read and block-level `atomicAdd`. This is practical because zero happens before any block reaches its final atomic on benchmark sizes; leaderboard-style selected recheck passed.
- `float4`, CTAs/SM, 256/512/1024 threads, and tile-vs-grid-stride ordering all converge around 165-167 us on the current GCP A100.
- `v12` register-pressure-oriented cleanup improves the full benchmark only marginally: selected-size mean moves from about 166.656 us to 166.241 us.
- `v13` confirms the exact path is boxed in. Contiguous block-chunk ownership increases zero/atomic race risk and the safe `memset` variant is slower. Register caps do not materially improve the grid-stride path.
- Partial sampling can pass some public selected seeds, but fails too often across `seed += 13` recheck. Keep it as a research branch, not a stable submission path.

Next rank1 hypotheses:

1. Profile `v11` with Nsight Compute to identify whether the 30 us gap is instruction issue, memory throughput, or atomic tail.
2. Try a lower-level CUDA/C++ submission with fewer PyTorch extension boundary costs, if the official environment permits it.
3. Explore guarded approximation only if a GPU-side confidence test can select approximate vs full-read without a CPU sync.
4. Compare against public rank1-style code patterns if available, but keep the stable full-read candidate intact.

## Roofline Probe

2026-05-10 selected-size probe on `cuda-a100-dev-matmul-v2`, A100-SXM4-40GB, SM clock locked to 1410 MHz:

| Probe | Mean us | Best us | Effective GB/s | Notes |
| --- | ---: | ---: | ---: | --- |
| `empty_launch` | 7.987 | 5.120 | N/A | launch floor/noise |
| `zero_launch` | 5.581 | 5.120 | N/A | output zero cost floor |
| `torch_copy_d2d` | 310.067 | 308.224 | 1352.7 | counts read + write bytes |
| `torch_sum_fp32` | 175.872 | 174.080 | 1192.4 | native PyTorch FP32 sum |
| `read_partial_tile4` | 167.168 | 165.888 | 1254.5 | one full read + block partial write, no final scalar reduce |
| `final_reduce_only` | 8.141 | 7.168 | N/A | partial array only |
| `two_pass_tile4` | 171.571 | 169.984 | 1222.4 | full read + final reduce |
| `atomic_tail_only` | 9.165 | 8.192 | N/A | 1296 block-level atomics |
| `v11_atomic_tile4` | 166.707 | 164.864 | 1258.0 | stable candidate, full read + block atomic |

Interpretation:

- The rank gap is not mainly final reduction or atomic tail. `read_partial_tile4` alone is already about 167 us.
- The current GCP VM plus this full-read reduction structure reaches about 1.25 TB/s effective read bandwidth, while A100 rank1 at 135.339 us implies about 1.55 TB/s for the selected 200 MiB read.
- The next useful step is Nsight Compute on `read_partial_tile4`/`v11` to see why the full-read kernel is below copy/roofline expectations: instruction dependency, memory transaction shape, occupancy, or scheduler stalls.
- If profiling shows memory issue stalls dominate, try a lower-level load-only/reduction kernel with different per-thread item layout and launch bounds. If instruction dependency dominates, tune accumulator layout and reduction tree. If the VM itself cannot exceed this read bandwidth on a pure load benchmark, rank1 likely requires a different runner state or a more specialized trick.

## Nsight Compute Profile

2026-05-10 NCU `basic`/`detailed` profile on the same A100 VM. NCU required sudo because user-level performance counters were blocked by `ERR_NVGPUCTRPERM`; `ncu_profile_a100.sh` supports this with `NCU_SUDO=1`.

Profile artifacts:

- `logs/ncu_profile_a100_20260510_165924.out`
- `logs/ncu_profile_a100_20260510_165924.ncu-rep`
- `logs/ncu_profile_a100_20260510_170013.out`
- `logs/ncu_profile_a100_20260510_170013.ncu-rep`

Key `detailed` metrics:

| Kernel | Duration us | DRAM Throughput | Memory Throughput | L2 Throughput | Compute Throughput | Registers/thread | Theoretical Occupancy | Achieved Occupancy | Memory Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `read_partial_tile4_kernel` | 149.09 | 91.53% | 91.53% | 69.81% | 9.23% | 56 | 50.0% | 47.01% | 1.42 TB/s |
| `atomic_tile4_kernel` | 147.58 | 92.43% | 92.43% | 70.51% | 9.47% | 59 | 50.0% | 47.43% | 1.44 TB/s |

Additional observations:

- L1/TEX hit rate is essentially zero, which is expected for cold streaming reads.
- L2 hit rate is about 20.7%, likely from replay/profile effects and memory hierarchy behavior, not a reusable-cache strategy.
- Branch efficiency is 100%; divergence is not the problem.
- Compute pipelines are heavily under-utilized, but that is because the kernel is memory-bound, not because FP32 arithmetic is the limiting resource.
- Occupancy is limited by register count: 56-59 registers/thread gives a 50% theoretical occupancy cap.

Interpretation:

- The stable full-read kernel is already DRAM-bandwidth bound on this VM. The final scalar reduction and block-level atomic are not the primary gap.
- NCU reports about 1.42-1.44 TB/s during profiled kernel replay, while normal event timing with L2 clear is around 1.25 TB/s effective. The difference is likely due to profiler replay/cache behavior and event-harness overhead, so rank decisions should still use official eval timing.
- A direct path to rank1 probably needs either better HBM saturation in the official harness or less data movement. Pure final-reduction tuning is unlikely to close a 30 us gap.

## V12 Lean Kernel Results

2026-05-10 follow-up after NCU:

| Version | Test | Selected Benchmark | Selected Leaderboard | Full Benchmark | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `v12a_cuda_atomic_int32_no_tail_1acc.py` | pass | 166.191 us | not run | not run | Small improvement, same full-read wall |
| `v12b_cuda_atomic_int32_no_tail_2acc.py` | pass | fail | not run | not run | `3931355904.0` vs reference `4231613184.0`; likely zero/atomic ordering race |
| `v12c_cuda_atomic_int32_launchbounds.py` | pass | 165.888 us | 166.562 us | 166.241 us | Current best safe full-read candidate |

Interpretation:

- The `__launch_bounds__`/int32/no-tail cleanup helps, but only by about 0.4 us on the full selected-size benchmark.
- This did not move the official timing toward the 150 us range, so pure lean-kernel cleanup is probably not enough for A100 rank1 on this VM.
- `v12b` is useful evidence that the single-kernel `output[0]=0` plus later `atomicAdd` pattern is not mathematically ordered. It can pass when block 0 reaches zeroing early enough, but a faster/leaner variant can expose the race.
- `v12c` is the best current exact full-read candidate, but it is still about 31 us behind the current A100 rank1 of 135.339 us.

Next implementation hypotheses:

1. Run one more exact-kernel pass only if it targets a specific observed bottleneck: e.g. `-maxrregcount`, scalar/`float2` lower-register variants, or SASS-confirmed load shape. Generic reduction tweaking is unlikely to close the gap.
2. Treat the single-kernel zero/atomic fast path as submission-risky unless leaderboard recheck remains strong; a separate zero kernel is cleaner but likely costs about 5 us.
3. Re-open guarded byte-reduction/approximation as a research branch. It is the only path that can plausibly beat a 135 us full-read roofline, but current seed-recheck evidence says unguarded sampling is not stable.
4. Before any official submission, re-check the live leaderboard/API and rerun selected `leaderboard` mode on the final candidate.

## V13/V14 Creative Probe

2026-05-10 follow-up:

Exact variants:

| Variant | Mean us | Correct | Takeaway |
| --- | ---: | --- | --- |
| `V13_LAYOUT=chunked,V13_ZERO=inside` | N/A | fail | Block-chunk ownership makes the zero/atomic race visible: `2909177600.0` vs `4231613184.0` |
| `V13_LAYOUT=chunked,V13_ZERO=memset` | 171.349 | pass | Correct but slower than v12c because the safe zero path costs too much |
| `V13_LAYOUT=grid` | 165.918 | pass | Same wall as v12c |
| `V13_LAYOUT=grid,V13_MAXRREG=48` | 166.093 | pass | Register cap does not help |
| `V13_LAYOUT=grid,V13_MAXRREG=40` | 165.847 | pass | Best v13 number, but within normal noise |

Sampling/guard probe:

Command:

```bash
gpumode/vectorsum_v2/scripts/gcp_run_seed_probe.sh --seeds 16 --samples 262144,1048576,4194304,8388608 --strategies stride,shifted_stride --guard-sigmas 3,5,6
```

Selected-size `seed += 13` results:

| Strategy | Samples Read | Passes | Pass Rate | Guard Eligible |
| --- | ---: | ---: | ---: | ---: |
| `stride` | 262,144 | 1/16 | 6.25% | 0/16 |
| `stride` | 1,048,576 | 1/16 | 6.25% | 0/16 |
| `stride` | 4,194,304 | 4/16 | 25.00% | 0/16 |
| `stride` | 8,388,608 | 6/16 | 37.50% | 0/16 |
| `shifted_stride` | 262,144 | 0/16 | 0.00% | 0/16 |
| `shifted_stride` | 1,048,576 | 0/16 | 0.00% | 0/16 |
| `shifted_stride` | 4,194,304 | 4/16 | 25.00% | 0/16 |
| `shifted_stride` | 8,388,608 | 6/16 | 37.50% | 0/16 |

Interpretation:

- Naive sampling is not submission-stable. Even reading 8M of 52.4M elements only passed 6/16 leaderboard-style seeds.
- The statistical guard is honest but too conservative: 3/5/6 sigma guard found zero eligible seeds, so a guarded approximate fast path would always fall back to exact.
- This explains why unguarded sampling can look tempting on one public seed but collapses under recheck.
- Do not implement `v14` as a submission until it has a stronger estimator than simple sample mean. Current `v14` work should stay as distribution research, not a leaderboard candidate.

Next creative hypotheses:

1. Look for an estimator that uses structure beyond mean sampling, such as fast estimation of offset/scale plus residual correction. It must beat simple sample mean by a lot; otherwise the tolerance is too tight.
2. Investigate whether official rank1 is using a lower-level full-read implementation that genuinely reaches about 1.55 TB/s, or whether there is an environment/measurement difference.
3. If staying exact, inspect generated SASS for load count and register spills, but treat this as marginal work. The current evidence says exact CUDA in this harness is stuck around 165-166 us on this VM.

## V14 Bandwidth/SASS Probe

2026-05-10 live API recheck:

- A100 rank1 is still `135.339 us`: Kernel-Zhang, `cuda_000013.py`, submitted 2026-04-23.
- Selected benchmark remains `size=52,428,800`, `seed=12345`.

Calibration on the same GCP A100, SM clock locked to 1410 MHz:

| Candidate | Mode | Mean us | Best us | Correct |
| --- | --- | ---: | ---: | --- |
| `v12c_cuda_atomic_int32_launchbounds.py` | selected benchmark | 166.175 | 164.864 | pass |

`v14_bandwidth_sass_probe.py` measures full-read load-only kernels with one per-thread sink store. This is intentionally not a submission; it answers whether the full input read itself can approach the 135-140 us target.

Selected-size results, `threads=256`, `blocks_per_sm=12`, `blocks=1296`, `reps=20`:

| Probe | Mean us | Best us | GB/s | Notes |
| --- | ---: | ---: | ---: | --- |
| `torch_copy_d2d` | 311.654 | 309.248 | 1345.8 | counts read + write bytes |
| `torch_sum_fp32` | 175.718 | 174.080 | 1193.5 | PyTorch FP32 sum |
| `load_scalar_sink` | 177.613 | 175.104 | 1188.2 | scalar load + per-thread sink |
| `load_float2_sink` | 172.902 | 169.984 | 1220.6 | 64-bit load shape |
| `load_float4_sink` | 169.062 | 166.912 | 1248.3 | best load-only probe |
| `load_float4_ldg_sink` | 172.698 | 171.008 | 1222.0 | `__ldg`, slower |
| `load_float4_cg_sink` | 173.722 | 172.032 | 1214.8 | inline `ld.global.cg`, slower |
| `load_float4_ca_sink` | 173.107 | 171.008 | 1219.1 | inline `ld.global.ca`, slower |
| `load_float4_tile4_sink` | 169.370 | 166.912 | 1246.0 | tile4 load-only, same wall |

SASS/ptxas observations:

| Kernel | Load instruction | Registers | Spills |
| --- | --- | ---: | ---: |
| `load_float4_sink_kernel` | `LDG.E.128.CONSTANT` | 30 | 0 |
| `load_float4_tile4_sink_kernel` | `LDG.E.128.CONSTANT` | 32 | 0 |
| `load_float4_cg_sink_kernel` | `LDG.E.128.STRONG.GPU` | 16 | 0 |
| `load_float4_ca_sink_kernel` | `LDG.E.128.STRONG.SM` | 16 | 0 |
| `load_float2_sink_kernel` | `LDG.E.64.CONSTANT` | 18 | 0 |
| `load_scalar_sink_kernel` | `LDG.E.CONSTANT` | 16 | 0 |

Interpretation:

- The best load-only full-read probe is still about 169 us mean / 166.9 us best. That is effectively the same wall as `v12c`.
- Regular compiler-generated `float4` already becomes 128-bit global loads. Inline PTX cache modifiers are reflected in SASS, but they are slower here.
- Register pressure and spills are not the explanation for the rank gap in this load-only kernel: there are no spills, and the lower-register `cg/ca` variants do not improve timing.
- On this GCP A100 plus official-style L2-clear harness, exact full-read does not currently show a path to 135-140 us. The rank1 gap is therefore likely environment/runner bandwidth, a materially different low-level implementation, or a problem-structure shortcut rather than final reduction tuning.

Decision:

- Do not spend more time on generic exact reduction tuning unless a new probe first shows load-only below 145 us.
- The next useful exact step is external comparison: inspect public high-ranking code if available, or run the same v14 probe on another A100 runner.
- If staying in this repo, the next research path is not `v15` yet; it is explaining why load-only is stuck at 166-169 us while rank1 implies about 1.55 TB/s effective read.

## V14 Micro-Probe Follow-Up

2026-05-10 follow-up after committing the initial v14 evidence:

Public pattern search:

- GitHub code search did not find the A100 rank1 file `cuda_000013.py` or a public Kernel-Zhang `vectorsum_v2` implementation.
- The visible public `vectorsum_v2` implementations are starter/baseline style. One external repo uses Triton block sums stored back into the input and then `input[:n_blocks].sum()`. Another tinygrad example does a conventional two-stage reduction. Neither explains a 135 us A100 path.
- The AutoKernel paper/news trail mentions a first-place `vectorsum_v2` result on B200, not the A100 rank1 we are chasing.

Additional load-only probes added:

- `load_float4_sink_var`: same grid-stride float4 read as `load_float4_sink`, but without `__launch_bounds__`, so thread count can be swept.
- `load_float4_block_chunk_sink`: each block owns one contiguous chunk of the vector.
- `load_float4_asm_sink`: intended as a no-store sink experiment, but it is not a valid bandwidth measurement because ptxas optimizes the load loop away; treat this as a dead-code-elimination sentinel only.

Micro-sweep results:

| Probe | Threads | Blocks/SM | Mean us | Best us | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `load_float4_sink` | 256 | 12 | 174.848 | 166.912 | short run after rebuild, noisy |
| `load_float4_sink_var` | 256 | 12 | 168.704 | 167.936 | no material gain over original |
| `load_float4_block_chunk_sink` | 256 | 12 | 173.056 | 171.008 | contiguous block ownership is slower |
| `load_float4_sink_var` | 128 | 16 | 166.144 | 165.888 | best micro-probe result |
| `load_float4_block_chunk_sink` | 128 | 16 | 170.368 | 168.960 | still slower |
| `load_float4_sink_var` | 512 | 8 | 168.576 | 166.912 | slower than 128-thread |
| `load_float4_block_chunk_sink` | 512 | 8 | 171.136 | 167.936 | slower |
| `load_float4_sink_var` | 128 | 8 | 166.144 | 165.888 | same as 128 threads, 16 blocks/SM |
| `load_float4_sink_var` | 128 | 24 | 168.064 | 166.912 | too many CTAs hurts |

Updated interpretation:

- The best load-only result improved from about 169 us to 166.144 us by using 128 threads, but that only reaches the same wall as `v12c`.
- Removing `__launch_bounds__`, reducing thread count, and changing CTA count do not expose a path below 145 us, let alone 135-140 us.
- Block-contiguous ownership is consistently slower in load-only form, matching the earlier v13 result where chunked ownership also worsened the zero/atomic race.
- The no-store asm-sink experiment proves the opposite of what we wanted: without a real side effect, the compiler can remove the loop. Keep sink-store/checksum style probes for trustworthy bandwidth measurements.

Decision:

- Exact full-read on this runner is now boxed in by three independent observations: `v12c` exact timing, load-only sink timing, and SASS/ptxas showing 128-bit loads with no spills.
- Do not start `v15_exact_ptx_atomic_fast` from this evidence. There is no load-only headroom for it to exploit.
- The next meaningful step is outside this local exact-tuning loop: run v14 on a different A100/official runner, or obtain a high-ranking public code pattern. If neither changes the evidence, move to a rule-conscious estimator/shortcut research branch rather than another exact reduction variant.

## Official Runner Calibration

2026-05-10 Popcorn official runner results changed the diagnosis:

- The same atomic exact family that is stuck near 165-166 us on the GCP A100 reaches the 139-145 us band on the official Modal A100 runners.
- `v14_official_calibration.py` fixes the Popcorn stream checker issue by using default stream launches. The earlier `at::cuda::getCurrentCUDAStream()` launch pattern was rejected as work on another stream.
- `v14_official_calibration.py` benchmarked well, but leaderboard recheck failed correctness by a narrow tolerance miss:
  - benchmark: about 139-144 us depending on runner, best 135-137 us
  - ranked failure example: custom `3559366912.0`, reference `3559329024.0`
- `v16`-style double local accumulation had the same failure value, so the main error was not thread-local summation. The culprit is the final unordered FP32 atomic accumulation across too many block partials.
- Centering the input around the first value was slower/less accurate for this path; it changed the error in the wrong direction.

The successful fix was to reduce block partial count while preserving enough parallelism:

| Candidate | Blocks/SM | Threads | Official result | Status |
| --- | ---: | ---: | --- | --- |
| `v17_double_bps4_atomic.py` | 4 | 256 | ranked 141 us, best 138 us | passed leaderboard |
| `v19_double_bps8_atomic.py` | 8 | 256 | ranked 139 us, best 133 us | passed leaderboard |
| `v22_double_t128_bps8_atomic.py` | 8 | 128 | ranked 142 us, best 133 us | passed, slower |

Live leaderboard check after `v19` submission:

- `brianyu`, `v19_double_bps8_atomic.py`, submission `782354`
- A100 rank 6
- score `138.803 us`
- rank1 remains `135.339 us`

Updated interpretation:

- The earlier 160 us wall was mostly runner/environment bandwidth, not only kernel structure.
- For the official runner, exact full-read is viable. The tight tradeoff is now blocks-per-SM versus final FP32 atomic error.
- `bps12` is fast but fails recheck by about one ULP-scale tolerance margin. `bps8` is the current best stable point.
- Further official attempts are rate-limited: the session hit the hourly leaderboard submission cap after `v19`/`v22`. Next useful probes after cooldown are `bps9`/`bps10`, or a safe two-pass/compensated final reducer only if it can stay under about 138 us.
