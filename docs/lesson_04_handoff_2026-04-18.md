# Lesson 04 Handoff

기준 날짜: `2026-04-18`

주제: **Matmul — memory hierarchy 4단계. FP32 naive → tiled → register blocking → FP16 Tensor Core**

## 1. 이번 세션에서 한 일

### 커널 (4 버전 + CPU ref)

[src/matmul.cu](/Users/xavier/dev/cudatraining/src/matmul.cu:1) 신규:

- **v1 naive**: 1 thread → 1 output element (global memory only)
- **v2 tiled**: 32×32 shared memory tile (AI = 8 FLOP/byte)
- **v3 register**: block 128×128, thread tile 8×8, BK=8 — register blocking (AI = 32 FLOP/byte)
- **v4 tensor**: WMMA API, block 64×64, 4 warps, FP16 입력 → FP32 축적

### 자동화 / 인프라

- [scripts/run_matmul_sweep.sh](/Users/xavier/dev/cudatraining/scripts/run_matmul_sweep.sh:1) — 4 크기 × 4 버전 = 16 runs
- [scripts/gcp_run_lesson04.sh](/Users/xavier/dev/cudatraining/scripts/gcp_run_lesson04.sh:1)
- [Makefile](/Users/xavier/dev/cudatraining/Makefile:1) 에 `matmul` / `run-matmul` 타겟 추가

### 이벤트

- 세션 중 spot VM 1회 preemption → 재시작 후 sweep 완료
- 16 runs 정상 종료

## 2. 산출물

- [results/remote/matmul_t4.csv](/Users/xavier/dev/cudatraining/results/remote/matmul_t4.csv:1) — 16 rows
- [results/lesson04-run-20260418-202556.log](/Users/xavier/dev/cudatraining/results/lesson04-run-20260418-202556.log:1)

## 3. 핵심 숫자

### TFLOPS by 버전 × 크기 (T4)

| N | v1 naive | v2 tiled | v3 register | v4 tensor |
|---:|---:|---:|---:|---:|
| 256  | 0.22 | 0.56 | **0.20** ⚠ | 1.51 |
| 512  | 0.47 | 0.75 | 0.95 | 3.71 |
| 1024 | 0.40 | 0.64 | 1.43 | 3.59 |
| 2048 | 0.40 | 0.83 | **2.05** | **7.93** |

Peak 기준:
- T4 FP32 peak = 8.1 TFLOPS
- T4 FP16 Tensor Core peak = 65 TFLOPS

v3 @ 2048 = **25% of FP32 peak**
v4 @ 2048 = **12% of TC peak**

## 4. 교훈 5가지

### (a) v3가 작은 N에서 v2보다 느림 — Occupancy trap

N=256: v3 = 0.20 TFLOPS (v2 = 0.56).

Block tile 128×128 이라, N=256 에서 블록 수 = 4개. T4의 40 SM 중 **36 SM 놀고 있음**. v2 (tile 32×32) 는 같은 N에서 64 블록 → 모든 SM 활용.

일반 원칙: 타일 클수록 AI 증가 = 좋음, **단 블록 수 ≥ 2 × SM 수**여야 함. 이걸 놓치면 오히려 느려진다.

LLM 서빙 함의: 큰 matmul (FFN) 은 문제없음. 하지만 small-batch decode 의 attention matmul 은 "작은 tile variant" 필요.

### (b) v3의 본질 — 스레드당 8×8 output 을 레지스터에 보유

Shared memory 에서 같은 값을 읽는 횟수: v2 대비 8x 감소.
Block 당 thread 수: 1024 → 256 (4배 감소) → SM당 거주 블록 수 증가 = occupancy 회복.
Block tile: 32×32 → 128×128 (16배) → AI 4배 증가.

N=2048 에서 2.05 TFLOPS. FP32 의 "실용적 상한" (cuBLAS SGEMM 의 3-4 TFLOPS) 에 근접.

### (c) v4 = 새로운 지붕 진입 — Tensor Core

한 mma_sync 명령 = 16×16×16 매트릭스 곱 = 4096 FMA. 약 8 cycle 에 완료.

같은 warp-cycle 대비 FP32 FMA 의 **16x 처리량**.

단순 WMMA 만으로 7.93 TFLOPS (peak 의 12%). 최적화된 CUTLASS 는 40-50 TFLOPS.

우리가 12%만 찍는 이유:
- Fragment load 가 swizzled layout 이 아님
- Double buffering (다음 타일 HBM→shared 로드 + 현재 타일 mma 오버랩) 없음
- Block tile 64×64 로 작음 (AI 제한)

이 셋을 다 구현하는 게 CUTLASS. 수개월 작업.

### (d) 정밀도 cost

| 버전 | max_abs_err @ N=1024 |
|---|---:|
| v1, v2, v3 (FP32) | 7.6e-5 |
| v4 (FP16 input) | **1.4e-2** — **180배** |

FP16 입력 → mantissa 10 bits → ~3 decimal 정밀도. LLM 추론은 허용, 학습은 BF16/FP32 혼합 필요.

이게 **AWQ, GPTQ, FP8 quantization 이 존재하는 이유**. 더 낮은 정밀도 → 더 빠른 TC → 모델 outputs 이 거의 동일하면 이득.

### (e) 4개 regime 의 전체 지도

```
v1  0.4 TFLOPS  FP32 memory-bound 나쁨 (L2 의존)
v2  0.8         FP32 memory-bound 좋음 (tiling)
v3  2.0         FP32 memory→compute 경계 (register blocking)
v4  7.9         FP16 Tensor Core 진입 (WMMA)
───────────────
cuBLAS HGEMM  ~40      FP16 TC 튜닝된 구현
CUTLASS 3.x   ~50+     FP16 TC 최적화 극한
```

**LLM inference 의 거의 전부는 맨 아래 두 줄에서 실행.** v1~v4 는 사다리.

## 5. 시각적 요약 — Roofline 감각

```
                    │       Tensor Core peak ~~~~~~~~~~~~
perf ▲              │
(TFLOPS)            │                                  v4 ●
                    │                                (7.9)
                    │      ← FP32 peak 8.1 ──────────────
                    │
                    │                         v3 ●
                    │                       (2.0)
                    │                 v2 ●
                    │              (0.8)
                    │      v1 ●
                    │    (0.4)
                    └────────────────────────────────────▶
                       low AI                       high AI
                       (memory-bound)          (compute-bound)
```

v3 까지는 HBM 대역폭 싸움, v4 에서 축이 수직으로 점프 (다른 지붕).

## 6. 다음 레슨 후보

### Next A: softmax — 실전 kernel fusion 첫 경험

Reduction (max + sum) + element-wise (exp, divide). 단독 커널 3개 vs 1개 fused 커널 비교.
Attention 의 심장. LLM 서빙 bottleneck 의 상당부분이 여기.

### Next B: Triton 포팅 — 같은 문제를 Triton 으로

v1~v4 (or softmax) 를 Triton 으로 재작성. "CUDA vs Triton: 추상화의 비용" 블로그 (Phase 1 M1-2 로드맵).
Triton 에서 autotuner 가 register blocking 을 얼마나 잘 찾는가 비교.

### Next C: vLLM good-first-issue — 실제 OSS 진입

M2-3 계획. 지금까지 쌓은 감각으로 vLLM 이슈 트래커 보고 작은 버그 하나 잡기.

### 추천: **Next A (softmax)** 또는 **Next B (Triton)**

- A: 실무적 직결, fusion 개념 첫 경험
- B: Phase 1 의 핵심 블로그 주제 (`Triton vs CUDA: 추상화의 비용`) 도달
- C: 아직 빠름 — softmax 수준의 fused kernel 감각이 있으면 vLLM 이슈 선택도 훨씬 안목 있게.

## 7. VM 상태

- `cuda-t4-dev-lesson02` **RUNNING** (세션 중 1회 preemption, 수동 재시작 후 완료)
- 유지

## 8. 한 줄 요약

matmul 에서 `memory-bound 4단계 + Tensor Core 축 점프` 를 몸으로 경험. **0.4 → 7.9 TFLOPS = 20x**. `peak 의 몇 %` 같은 표현을 이제 숫자와 함께 읽고 말할 수 있음.
