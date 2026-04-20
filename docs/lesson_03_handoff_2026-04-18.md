# Lesson 03 Handoff

기준 날짜: `2026-04-18`

주제: **Reduction — shared memory, warp, shuffle. 병렬성의 첫 장벽**

## 1. 이번 세션에서 한 일

### 새 커널 (5개)

[src/reduction.cu](/Users/xavier/dev/cudatraining/src/reduction.cu:1) 신규 작성:

- **v1 atomic**: 모든 스레드가 하나의 global 주소에 `atomicAdd` (baseline, 나쁜 예)
- **v2 shared**: 블록당 shared memory 트리, 루트만 atomic
- **v3 unroll**: v2 + 마지막 warp 구간을 shuffle로 대체 (syncthreads 5번 제거)
- **v4 shuffle**: 각 warp가 shuffle로 로컬 리듀스 → warp 간에만 shared memory
- **thrust**: `thrust::reduce` (cub 기반 multi-pass, 비교 baseline)

### 자동화

- [scripts/run_reduction_sweep.sh](/Users/xavier/dev/cudatraining/scripts/run_reduction_sweep.sh:1) — 5 sizes × 5 versions = 25 runs (v1은 큰 n에서 iterations 축소)
- [scripts/gcp_run_lesson03.sh](/Users/xavier/dev/cudatraining/scripts/gcp_run_lesson03.sh:1) — copy → build → sweep → pull CSV
- [Makefile](/Users/xavier/dev/cudatraining/Makefile:1) 에 `reduction` / `run-reduction` 타겟 추가

### 실행

- 같은 VM `cuda-t4-dev-lesson02` 재사용 (T4, us-east1-d)
- 전체 25 runs 완료

## 2. 산출물

- [results/remote/reduction_t4.csv](/Users/xavier/dev/cudatraining/results/remote/reduction_t4.csv:1) — 25 rows
- [results/lesson03-run-20260418-160904.log](/Users/xavier/dev/cudatraining/results/lesson03-run-20260418-160904.log:1)

## 3. 가공된 핵심 숫자

### 버전별 best_ms (T4, block_size=256)

| n | v1 atomic | v2 shared | v3 unroll | v4 shuffle | thrust |
|---|---:|---:|---:|---:|---:|
| 2^20 | 2.082 | 0.018 | 0.013 | **0.012** | 0.028 |
| 2^22 | 8.309 | 0.070 | 0.066 | **0.066** | 0.089 |
| 2^24 | 33.227 | 0.260 | 0.258 | **0.258** | 0.289 |
| 2^26 | 132.903 | 1.089 | **1.087** | 1.090 | 1.118 |
| 2^28 | 531.577 | 4.860 | 4.870 | 4.887 | **4.648** |

### 유효 대역폭 (GB/s, n=2^22 기준)

| 버전 | GB/s | % of theoretical (320 GB/s) |
|---|---:|---:|
| v1 atomic | 2.0 | 0.6% |
| v2 shared | 240.9 | 75% |
| v3 unroll | 253.5 | 79% |
| v4 shuffle | 254.5 | 80% |
| thrust | 188.7 | 59% |

## 4. 세 가지 교훈

### (a) v1 atomic의 "2 GB/s floor"

v1의 유효 대역폭이 **n에 무관하게 ~2.0 GB/s 고정**. 이건 HBM 대역폭이 아니라 **atomic throughput 상한**이다. 100만 스레드가 하나의 주소에 동시 접근하면 하드웨어가 직렬화 — 실질적으로 1개 스레드가 일하는 것과 가까워짐.

교훈: atomic은 "결과 1개를 만드는 장치"로 쓰면 안 된다. **블록당 1번만** (트리 리덕션의 루트) 이나, **저빈도 카운터** 수준에서만 써야 함.

### (b) v2 → v3 → v4 progression at small n

작은 n(2^20)에서 각 최적화의 기여를 분리해서 볼 수 있다:

| 구간 | Δ time | 개선 % | 제거된 비용 |
|---|---|---|---|
| v1 → v2 | 2.082 → 0.018 ms | 113x | 직렬 atomic → 병렬 트리 |
| v2 → v3 | 0.018 → 0.013 ms | -29% | 마지막 warp의 `__syncthreads` 5번 |
| v3 → v4 | 0.013 → 0.012 ms | -6% | Shared memory → 레지스터 shuffle |

**v2→v3의 29%**가 특히 주목할만. `__syncthreads()` 가 얼마나 비싼지 — 한 번에 수백 cycle, 5번이면 수천 cycle. 작은 커널에서 이게 실행시간의 1/3을 차지한다.

### (c) 큰 n에선 모두 수렴

n ≥ 2^24 부터 v2/v3/v4/thrust 가 거의 동일 (±5%).

이유: HBM 대역폭이 **바닥** (~77% of theoretical)에 닿아있다. 읽어야 하는 바이트가 고정이고, 이걸 읽는 시간이 전체를 지배. 리덕션 트리를 어떻게 구성하든 "데이터 가져오는 시간"은 안 바뀜.

함의: **작은 n에서만 tail 최적화 신경 써라.** 큰 n이라면 "코드 단순성 > 커널 내부 미세 최적화".

## 5. Thrust 분석

| n | thrust / v4 비율 |
|---|---:|
| 2^20 | 2.3x **(느림)** |
| 2^22 | 1.35x 느림 |
| 2^24 | 1.12x 느림 |
| 2^26 | 1.03x 느림 |
| 2^28 | **0.95x (빠름)** |

Thrust는 cub 기반 **multi-pass 리덕션**: 1차로 블록당 부분합을 임시 버퍼에 쓰고, 2차 커널로 최종 합산. 오버헤드가 있어서 작은 n에 불리, 큰 n에서 HBM 스케줄링이 유리해서 역전.

**정확도는 thrust가 항상 최고** — 트리 깊이가 더 균형 잡혀서 부동소수점 오차 상쇄 유리.

**실무 판단**: 프로덕션 코드에서는 thrust/cub을 쓰고, 커스텀 커널은 n 특성이 명확하고 성능 차이가 의미있을 때만.

## 6. 왜 직접 짜보는가

프로덕션에서 thrust/cub을 쓰는 게 정답이라면, 왜 v1~v4를 손으로 짤까?

**시스템 판단자** 관점에서:
- "왜 atomic이 100배 느린가"를 **숫자로** 기억하는 엔지니어와 그렇지 못한 엔지니어의 판단력 차이
- Triton, Mojo, 새로운 언어/컴파일러가 나올 때마다 "저 언어가 이 레이어에서 뭘 숨기고 뭘 노출하는가"를 평가할 기준점
- fusion 결정: "이 커널 혼자 돌리면 80% bandwidth, 여기서 추가 이득은 다른 연산과 합치는 것뿐" 이런 판단을 자동으로 할 수 있어야 함

## 7. 다음 레슨 후보

### Next A: matmul — 진짜 compute-bound 영역

- 지금까지 vector_add, reduction 모두 memory-bound
- matmul은 O(N^3) compute vs O(N^2) memory → compute-bound 가능
- 단계:
    - v1 naive (global memory per access)
    - v2 tiled shared memory
    - v3 register blocking
    - v4 Tensor Core (WMMA API)
    - cuBLAS baseline
- 이게 **Phase 1 M2-3** 에 해당 ([phase1_plan.md](/Users/xavier/dev/cudatraining/docs/phase1_plan.md))

### Next B: softmax — serving에서 실제 쓰이는 커널

- reduction (max, sum) + element-wise 조합
- attention의 핵심 구성요소
- vLLM/flash-attn 에서 실제로 어떻게 구현되는지 읽을 동기 부여

### Next C: CUDA streams + async copy overlap

- Lesson 02 에서 보류했던 주제
- 지금은 H2D → kernel → D2H 직렬. 스트림으로 overlap 얼마나 빠져나오는지.

### 추천: **Next A (matmul)**

이유:
- Phase 1 로드맵의 핵심
- Tensor Core 가 LLM 서빙의 실제 주역
- memory-bound → compute-bound 로의 패러다임 전환을 몸으로 경험
- 포트폴리오 측면에서도 "matmul 다뤄봤다"가 가장 신호 강함

## 8. 다음 레슨 체크리스트 (Next A 선택 시)

1. `src/matmul.cu` — naive → tiled → register blocked
2. cuBLAS baseline (`cublasSgemm`) 비교
3. WMMA API 로 Tensor Core 활용 (FP16 input → FP32 accum)
4. Roofline plot: compute-bound 영역 진입 확인
5. `results/matmul_t4.csv`

## 9. VM 상태

`cuda-t4-dev-lesson02` : **RUNNING**. 유지.

## 10. 한 줄 요약

Lesson 03 성공. `atomic이 100x 느린 이유`, `__syncthreads 제거가 29% 기여`, `shuffle이 6% 추가`, `큰 n은 전부 수렴` — 이 4가지가 머릿속에 수치로 남음. 다음은 matmul로 **compute-bound 첫 접촉**.
