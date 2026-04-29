# Lesson 05 Handoff

기준 날짜: `2026-04-18`

주제: **Softmax & Fusion — 3-kernel naive vs fused vs online. Flash Attention 의 수학적 뼈대.**

## 1. 이번 세션에서 한 일

### 커널 3 버전

[src/softmax.cu](/Users/xavier/dev/cudatraining/src/softmax.cu:1) 신규:

- **v1 naive**: 3 개 커널 순차 (max → sum → divide). HBM 4 trips/elem.
- **v2 fused**: 1 커널, shared memory 에 행 캐싱. HBM 2 trips/elem.
- **v3 online**: 1 커널 + normalize pass. 행 캐싱 안 함 (임의 N 지원). HBM 3 trips/elem.

### 공용 primitive

- `warp_reduce_max`, `warp_reduce_sum` (shuffle 기반)
- `warp_reduce_online`: (max, sum) tuple 의 online 병합 공식 구현 — **Flash Attention 의 핵심 수식**

### 자동화

- [scripts/run_softmax_sweep.sh](/Users/xavier/dev/cudatraining/scripts/run_softmax_sweep.sh:1) — 4 N × 3 버전
- [scripts/gcp_run_lesson05.sh](/Users/xavier/dev/cudatraining/scripts/gcp_run_lesson05.sh:1)
- [Makefile](/Users/xavier/dev/cudatraining/Makefile:1) 에 `softmax` / `run-softmax` 타겟

## 2. 산출물

- [results/remote/softmax_t4.csv](/Users/xavier/dev/cudatraining/results/remote/softmax_t4.csv:1) — 12 rows
- [results/lesson05-run-20260418-211029.log](/Users/xavier/dev/cudatraining/results/lesson05-run-20260418-211029.log:1)

## 3. 핵심 숫자

### best_ms × effective GB/s (T4, M=4096)

| N | v1 ms | v1 GB/s | v2 ms | v2 GB/s | v3 ms | v3 GB/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.293 | 229 | **0.145** | 231 | 0.141 | 356 |
| 2048 | 0.530 | 253 | **0.285** | 236 | 0.279 | 361 |
| 4096 | 1.067 | 252 | **0.556** | 241 | 0.764 | 264 |
| 8192 | 2.212 | 243 | **1.470** | 183 ← dip | 1.669 | 241 |

Theoretical HBM = 320 GB/s. v3 의 356/361 GB/s 는 **L2 cache hit** 으로 이론 초과.

### 정확도
max_abs_err < 1e-6 (모든 조합). FP32 + __expf 로 충분한 정밀도.

## 4. 네 가지 교훈

### (a) Fusion 약속은 관찰 가능 — v2 가 **정확히 2x** (4096 까지)

이론 HBM trips: v1=4, v2=2 → v2 가 2x 빨라야.

관찰: 2.02x (N=1024), 1.86x (N=2048), 1.92x (N=4096). 거의 이론치.

**LLM inference 최적화의 절반 이상이 fusion 인 이유**가 이 숫자에 있음.

### (b) Occupancy cliff 가 N=8192 에서 나타남

v2 smem usage = N × 4 바이트.

| N | smem/block | blocks/SM (T4 64KB) | threads/SM | occupancy |
|---:|---:|---:|---:|---:|
| 1024 | 4 KB | 16 | 1024 | 100% |
| 4096 | 16 KB | 4 | 1024 | 100% |
| 8192 | 32 KB | **2** | **512** | **50%** |

N=8192 에서 v2 speedup 이 2x → 1.5x 로 떨어짐. 유효 GB/s 도 57% 로 저하 = **bandwidth 가 부족한 게 아니라 latency hiding 할 warp 가 부족**한 것.

**이게 shared memory 많이 쓰는 커널의 보편적 함정.**

### (c) L2 cache 가 v3 의 "extra read" 를 부분 흡수

이론: v3 는 3 trips (2 read + 1 write), v2 보다 1.5x 느려야.

관찰 N=1024: v3=0.141ms, v2=0.145ms. **v3 가 약간 더 빠름.**

이유:
- 행 = 4KB → L2 (4MB) 에 여유 있음
- Pass 1 의 입력이 pass 2 에서 **L2 hit**
- 유효 GB/s = 356 (theoretical 의 111%) — L2 가 흡수해서 HBM 이 아니라는 증거

N 이 커지면 (8192) L2 밀려나가며 혜택 감소, v2 가 재우세.

### (d) v3 의 진짜 가치 = **Flash Attention 의 math half**

v3 자체가 우리 크기에선 v2 보다 빠른 게 아님. 하지만:
- **임의의 N 에 대응 가능** (v2 는 N > 12288 에서 실패)
- 더 중요하게, online update 공식:
  ```
  new_max = max(m1, m2)
  new_sum = s1 * exp(m1 - new_max) + s2 * exp(m2 - new_max)
  ```
  은 **Flash Attention 의 attention output 업데이트 공식과 동일**
- FA 는 여기에 tiled matmul fusion 을 얹어 attention 중간 행렬 (P = softmax(Q@K^T)) 을 **HBM 에 내리지 않음**

**v3 을 구현 = FA 의 수학적 절반 이해.** 나머지 절반은 attention-specific tiling.

## 5. Regime 지도

```
regime                 주인공    왜
────────────────────────────────────────────────────────────
작은 N (smem 여유)      v2       fusion 완승 (2x)
중간 N (L2 히트)        v2 ≈ v3  L2 가 v3 의 3번째 read 흡수
큰 N (smem 포화)        v3 / FA  v2 occupancy 붕괴
매우 큰 N (attention)   FA       중간 행렬 materialize 불가
```

## 6. LLM serving 번역

**Decode phase** (sequence 생성 단계): seq_len 짧음 → v2 같은 단순 fused softmax 로 충분. 병목은 KV cache HBM 로드.

**Prefill phase** (초기 긴 context 처리): seq_len 수천 ~ 수만. Attention score 행렬이 거대 → **Flash Attention 필수**. 우리 v3 online 수식이 FA 의 뼈대.

## 7. 다음 단계 후보

### Next A: Triton 포팅 (Phase 1 M1-2 로드맵)

지금까지 쌓인 커널 (reduction / matmul / softmax) 을 Triton 으로 재작성. 블로그 "Triton vs CUDA: 추상화의 비용" 완성.

신호 가치 큼. 환경 세팅 약간 (Python, PyTorch, Triton 설치).

### Next B: Flash Attention 직접 구현

오늘 v3 가 절반 이해, matmul 레슨이 타일링 완료 → 둘 결합하면 FA-like 커널 가능.
- Q @ K^T 를 타일로 계산
- 각 타일에서 online softmax 업데이트
- 타일에서 @ V 바로 곱해서 output 누적

고난도. 3-5시간. 포트폴리오에서 가장 강한 한 블록.

### Next C: vLLM good-first-issue

M2-3 점프. 지금 시점이면 충분히 이슈 선택 안목 있음 (fusion, KV cache, attention 다 이해).

### 추천

**A (Triton) 를 먼저**. 이유:
1. Phase 1 로드맵의 대표 블로그 주제
2. Triton 이 내부적으로 뭘 숨기는지 = 오늘 우리가 손으로 짠 optimization (online reduce, fusion, tiling) 을 컴파일러가 자동 결정. 직접 비교 가능.
3. B (Flash Attention) 는 Triton 에서 짜는 게 CUDA 보다 훨씬 편함 — Triton 포팅 경험 있으면 B 가 쉬워짐.

## 8. VM 상태

`cuda-t4-dev-lesson02` : RUNNING. 유지.

## 9. 한 줄 요약

Softmax 에서 **fusion 2x** 를 숫자로 확인, **occupancy cliff** 와 **L2 masking** 이라는 커널 튜닝의 두 패턴을 목격, 그리고 **Flash Attention 의 math half** 인 online softmax 를 직접 구현. CUDA 의 레벨 1 은 대체로 이 지점까지.
