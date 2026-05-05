# GPUMODE matmul_v2 L4 Plan

기준 날짜: 2026-05-05

이 문서는 GPUMODE `matmul_v2` 리더보드에서 L4 기준 좋은 성적을 내기 위한 실행 계획이다. 핵심 전략은 단순하다. 먼저 `torch.mm(a, b, out=c)`로 안전하고 강한 기준선을 만들고, 그 다음 고정 benchmark shape에서만 Triton/CUTLASS 계열 커널을 하나씩 붙여서 이기는 shape만 남긴다.

## 목표

- 1차 목표: L4에서 correctness가 안전한 `v0_safe` 제출을 만든다.
- 2차 목표: 큰 3개 shape만 Triton으로 대체한 `v1_hybrid_large`를 측정한다.
- 3차 목표: 중간 shape까지 확장한 `v2_hybrid_mid`에서 2.25 ms 아래를 노린다.
- 상위권 목표: 2.20 ms 근처 또는 그 아래. 2026-05-05 GPUMODE API 기준 L4 1위는 `2200.917 us`다.

## 현재 리더보드 스냅샷

출처: `https://www.gpumode.com/api/leaderboard/540`, 2026-05-05 확인. 아래 표는 `v4_bigshape_cache_hint.py` 제출 이후 최신 스냅샷이다.

| Rank | Score | User | File | Submitted |
| ---: | ---: | --- | --- | --- |
| 1 | 2076.331 us | brianyu | `v4_bigshape_cache_hint.py` | 2026-05-05 |
| 2 | 2200.917 us | iharryli | `ori_submission.py` | 2026-03-03 |
| 3 | 2226.859 us | mreso | `solution.py` | 2026-03-04 |
| 4 | 2255.189 us | rajesh0042 | `matmul_v5.py` | 2026-03-13 |
| 5 | 2302.197 us | burtenshaw | `submission_team_mm_r2_matmul_out.py` | 2026-03-02 |

이 숫자는 계속 변할 수 있으므로, 제출 전에는 반드시 같은 API나 리더보드 페이지에서 다시 확인한다.

## 문제 요약

공식 reference는 `problems/pmpp_v2/matmul_py` 기준으로 `a @ b`를 반환한다. 입력은 `(a, b, c)`이고 dtype은 `float16`이다.

```python
def ref_kernel(data):
    with DeterministicContext():
        a, b, c = data
        return a @ b
```

공식 `task.yml`의 public tests와 benchmark는 분리되어 있다.

Public tests:

- `(64, 64, 64)`
- `(128, 128, 128)`
- `(256, 256, 256)`
- `(32, 512, 32)`
- `(64, 1024, 64)`

Benchmarks:

| Shape `(M, N, K)` | FLOP | Share | Dense 121 TFLOP/s lower bound | First action |
| --- | ---: | ---: | ---: | --- |
| `(128, 128, 128)` | 0.004 GF | 0.002% | 0.035 us | Keep torch first |
| `(256, 256, 256)` | 0.034 GF | 0.015% | 0.277 us | Keep torch first |
| `(512, 512, 512)` | 0.268 GF | 0.122% | 2.218 us | Compare later |
| `(1024, 1024, 1024)` | 2.147 GF | 0.974% | 17.748 us | Try 64x128x64 |
| `(2048, 2048, 2048)` | 17.180 GF | 7.794% | 141.982 us | Try 128x128x64 |
| `(1024, 1536, 1024)` | 3.221 GF | 1.461% | 26.622 us | Try 64x128x64 |
| `(2048, 3072, 2048)` | 25.770 GF | 11.691% | 212.974 us | Try 128x128x64 |
| `(4096, 5120, 4096)` | 171.799 GF | 77.940% | 1419.824 us | First priority |

Total benchmark work is `220.423 GFLOP`. NVIDIA's public L4 spec lists FP16 Tensor Core `242 TFLOP/s` with sparsity and says non-sparse specs are one half lower, so dense FP16 peak is treated as about `121 TFLOP/s`. That gives a theoretical compute lower bound around `1.82 ms`. L4 bandwidth is `300 GB/s`, but these shapes are compute dominated if tiling is good.

## Operating Principle

Do not replace PyTorch globally. Replace only a shape that has both:

1. Correctness pass against the official reference.
2. Consistent speedup versus `torch.mm(a, b, out=c)`.

If a custom kernel is equal, noisy, or only faster in local timing but slower on GPUMODE, keep the torch path. On this problem, an unfancy correct `out=` path is already a serious baseline.

## v0_safe

First submission:

```python
#!POPCORN leaderboard matmul_v2
#!POPCORN gpus L4

import torch
from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    a, b, c = data
    torch.mm(a, b, out=c)
    return c
```

Why this matters:

- It avoids `tmp = a @ b` followed by a copy into `c`.
- It uses the same PyTorch/cuBLAS path family as the reference.
- It should pass public tests and benchmark correctness with very low risk.
- It tells us how much room is actually left against current L4 leaders.

Definition of done:

- GPUMODE test passes.
- GPUMODE benchmark score and per-shape means are recorded.
- If leaderboard score is already near `2.30 ms`, proceed to shape-specific replacement.

## v1_hybrid_large

Only replace the high-impact shapes:

- `(2048, 2048, 2048)`
- `(2048, 3072, 2048)`
- `(4096, 5120, 4096)`

Candidate Triton config:

| Name | BM | BN | BK | Warps | Stages | GROUP_M | Use |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| C | 128 | 128 | 64 | 4 | 4 | 8 | Default large |
| D | 128 | 128 | 32 | 4 | 5 | 8 | If BK=64 hurts correctness or occupancy |
| E | 128 | 256 | 64 | 8 | 4 | 8 | Only for biggest shape |

Hot path requirements:

- No masks for benchmark shapes.
- Shape dispatch should check exact `(M, N, K)` tuples.
- Non-benchmark tests must fall back to `torch.mm(out=c)`.
- Store `acc.to(tl.float16)` into provided `c`.

Initial dispatch:

```python
BENCHMARK_SHAPES = {
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (1024, 1536, 1024),
    (2048, 3072, 2048),
    (4096, 5120, 4096),
}

LARGE_TRITON = {
    (2048, 2048, 2048),
    (2048, 3072, 2048),
    (4096, 5120, 4096),
}

def custom_kernel(data):
    a, b, c = data
    m, k = a.shape
    n = b.shape[1]
    shape = (m, n, k)

    if shape not in BENCHMARK_SHAPES:
        torch.mm(a, b, out=c)
        return c

    if shape in LARGE_TRITON:
        # launch 128x128x64 no-mask kernel
        return c

    torch.mm(a, b, out=c)
    return c
```

Definition of done:

- Public tests pass.
- Each large shape passes benchmark correctness.
- Any large shape that does not beat torch by at least about `0.5%` after repeated runs is removed from `LARGE_TRITON`.

## v2_hybrid_mid

Add medium shapes after the large ones are stable:

- `(1024, 1024, 1024)`
- `(1024, 1536, 1024)`
- optional `(512, 512, 512)` only after the two 1024 cases are understood

Candidate configs:

| Name | BM | BN | BK | Warps | Stages | GROUP_M | Rationale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A | 64 | 64 | 64 | 4 | 4 | 8 | More CTAs, small/medium saturation |
| B | 64 | 128 | 64 | 4 | 4 | 8 | First medium candidate |
| C | 128 | 128 | 64 | 4 | 4 | 8 | Compare against large default |

The 1024 square shape has only 64 CTAs with `128x128`, so `64x128` can win despite less per-CTA work. Measure, do not guess.

## v3_full_dispatch

Only after v1/v2:

- Try `(128, 128, 128)` and `(256, 256, 256)` with Triton only if launch overhead and cache behavior beat cuBLAS.
- For these small shapes, one extra Python branch can matter less than one extra allocation, but Triton launch overhead can also dominate. Keep the faster path, even if it is boring.

Expected result:

- Small shapes probably stay `torch.mm(out=c)` unless a specialized tiny kernel wins.
- Overall score movement should come mostly from the final two shapes.

## v4_cutlass_or_cublaslt

If Triton cannot beat PyTorch on large shapes:

- Try direct cublasLt dispatch if allowed by the environment.
- Try CUTLASS/CuTe-style generated kernels if packaging is allowed and compilation time fits the leaderboard.
- Compare against a manually selected cuBLAS algorithm only if the submission path can call it without unacceptable overhead.

This is not the first move. It is the escape hatch after the simple hybrid strategy has produced honest numbers.

## Benchmark Discipline

Official eval behavior that matters:

- Benchmark mode warms up using the first benchmark.
- Each timed run calls `clear_l2_cache()` before CUDA event timing.
- Leaderboard mode regenerates input data during timed loops and rechecks correctness.
- CUDA event timing measures only the submitted function region, but Python dispatch and kernel launches inside it still affect the event interval when they enqueue work.

Rules:

- Record `best`, `mean`, `std`, `err`, and `runs` per benchmark.
- Trust GPUMODE numbers over local numbers.
- Prefer changes with stable improvements across at least two benchmark submissions.
- Do not chase 128/256 tiny shapes until the big shape path is stable.

## Correctness Risks

Triton `tl.dot` with FP32 accumulation is mathematically sensible, but it is not guaranteed to round identically to PyTorch/cuBLAS. The official checker uses `verbose_allclose` with default `rtol=1e-5` and `atol=1e-8`, so shape-level correctness is a real gate.

Risk controls:

- Keep `torch.mm(out=c)` fallback for all non-benchmark shapes.
- Add custom kernels one shape group at a time.
- If one shape fails correctness, remove only that shape from dispatch.
- Do not use approximate modes that change numerical behavior unless verified.
- Preserve dtype and contiguity assumptions from `generate_input`.

## Experiment Log Template

Use this table for every submission:

| Version | Enabled custom shapes | Score us | Rank | Correct | Notes |
| --- | --- | ---: | ---: | --- | --- |
| v0_safe | none | TBD | TBD | TBD | `torch.mm(out=c)` |
| v1_large_C | 2048 sq, 2048x3072, 4096x5120 | TBD | TBD | TBD | 128x128x64 |
| v1_large_D | same | TBD | TBD | TBD | 128x128x32 |
| v2_mid_B | v1 winners + 1024 cases | TBD | TBD | TBD | 64x128x64 |

Per-shape table:

| Version | Shape | torch us | custom us | Delta | Keep? | Failure mode |
| --- | --- | ---: | ---: | ---: | --- | --- |
| TBD | `(4096, 5120, 4096)` | TBD | TBD | TBD | TBD | TBD |

## Work Plan

### Phase 0: Baseline

- [x] Create this plan.
- [ ] Create `gpumode/matmul_v2/submissions/v0_safe.py`.
- [ ] Submit test mode on L4.
- [ ] Submit benchmark mode on L4.
- [ ] Record score and per-shape stats in this README.

### Phase 1: Local/remote harness

- [ ] Add a small parser for GPUMODE benchmark output if logs are easy to export.
- [ ] Keep a local `experiments/` note or CSV with version, score, and per-shape timings.
- [ ] Confirm exact package versions in the L4 runner when visible.

### Phase 2: Large Triton kernel

- [ ] Implement a no-mask Triton GEMM kernel in `v1_hybrid_large.py`.
- [ ] Dispatch only the large 3 benchmark shapes.
- [ ] Submit config C.
- [ ] Submit config D only if C is close or unstable.
- [ ] Try config E only on `(4096, 5120, 4096)`.
- [ ] Remove losing shapes from custom dispatch.

### Phase 3: Medium shapes

- [ ] Add config B for `(1024, 1024, 1024)` and `(1024, 1536, 1024)`.
- [ ] Compare config A, B, and C.
- [ ] Keep only shape/config pairs that beat v1.

### Phase 4: Small shapes

- [ ] Try tiny specialized Triton only for `(128, 128, 128)`, `(256, 256, 256)`, and maybe `(512, 512, 512)`.
- [ ] Keep torch path if custom launch overhead wins nothing.

### Phase 5: Escalation

- [ ] Investigate cublasLt or CUTLASS/CuTe only if Triton cannot improve the final score.
- [ ] Stop adding complexity once score movement is below measurement noise.

## Immediate Next Step

The active L4 ranked submission is `submissions/v4_bigshape_cache_hint.py`, which reached L4 rank 1 at `2076.331 us`.

The active A100 ranked submission is `submissions/a100/a100_v0_safe.py`, which reached A100 rank 2 at `634.197 us` with submission `780614`. The gap to A100 rank 1 is `4.437 us` (`0.70%`), so the next A100 work should be a narrow rank-1 attack:

1. Keep `a100_v0_safe.py` as the active ranked submission; `a100_v1_aten_mm_out.py` was benchmarked and ranked after cooldown but scored worse (`660 us`, submission `780623`).
2. Investigate whether direct cuBLAS/cuBLASLt can remove PyTorch dispatch overhead or force the best GEMM algorithm.
3. Try a minimal compiled extension only if the GPUMODE environment allows it without setup overhead inside the timed region.
4. Avoid broad Triton tile sweeps unless a new kernel family can plausibly beat cuBLAS on A100.

## Sources

- GPUMODE leaderboard page: <https://www.gpumode.com/leaderboard/540?tab=rankings>
- GPUMODE leaderboard API: <https://www.gpumode.com/api/leaderboard/540>
- Reference kernels repo: <https://github.com/gpu-mode/reference-kernels/tree/main/problems/pmpp_v2/matmul_py>
- NVIDIA L4 official specs: <https://www.nvidia.com/en-us/data-center/l4/>
- PyTorch `torch.mm` docs: <https://docs.pytorch.org/docs/stable/generated/torch.mm.html>
