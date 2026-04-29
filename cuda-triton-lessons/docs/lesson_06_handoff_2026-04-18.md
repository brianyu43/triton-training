# Lesson 06 Handoff — Flash Attention (Capstone)

기준 날짜: `2026-04-18`

주제: **Attention 의 두 구현 — naive 3-kernel pipeline 대 Flash Attention v1 (tile + online softmax fusion).**

## 1. 이번 세션에서 한 일

### 두 구현

[src/flash_attention.cu](/Users/xavier/dev/cudatraining/src/flash_attention.cu:1) 신규:

- **naive**: 3 개 커널 — `attn_naive_qk_scale` → `attn_naive_softmax` → `attn_naive_pv`. 중간 행렬 S, P (N×N) 를 HBM 에 완전 materialize. 레슨 4 (tiled matmul) + 레슨 5 v2 (fused softmax) 를 그대로 체인.
- **flash**: 단일 커널 `flash_attention_v1`. Q 를 Br=64 행 블록, K/V 를 Bc=32 열 블록으로 타일링. 각 타일마다:
  1. Q @ K^T 타일 계산 (registers)
  2. Online softmax 로 running (m, l) 업데이트 — 레슨 5 v3 의 수식
  3. O 를 rescale 후 P @ V 누적
  4. 타일 버림. **S/P 를 HBM 에 안 씀.**

### 버그 한 번 잡음

첫 런에서 naive 의 `max_abs_err = 0.05` → Q@K^T 타일 로딩에서 K 행 인덱스를 `threadIdx.x` 로 써서 K tile 이 tx 에만 의존하게 됨 (diagonal 값만 올바름). `threadIdx.y` 로 행 인덱스 수정 → 정확도 복구 (max_rel_err < 0.015).

### 자동화

- [scripts/run_flash_attention_sweep.sh](/Users/xavier/dev/cudatraining/scripts/run_flash_attention_sweep.sh:1) — N ∈ {512, 1024, 2048, 4096} × {naive, flash}
- [scripts/gcp_run_lesson06.sh](/Users/xavier/dev/cudatraining/scripts/gcp_run_lesson06.sh:1)
- [Makefile](/Users/xavier/dev/cudatraining/Makefile:1) 에 `flash_attention` / `run-flash-attention` 타겟

## 2. 산출물

- [results/remote/flash_attention_t4.csv](/Users/xavier/dev/cudatraining/results/remote/flash_attention_t4.csv:1) — 8 rows
- [results/lesson06-run-20260418-215341.log](/Users/xavier/dev/cudatraining/results/lesson06-run-20260418-215341.log:1)

## 3. 핵심 숫자

### best_ms, GFLOP/s, HBM bytes (T4, d=64, FP32)

| N | naive ms | flash ms | **speedup** | naive HBM (MB) | flash HBM (MB) | **HBM ratio** |
|---:|---:|---:|---:|---:|---:|---:|
| 512  | 0.402 | 0.593 | 0.68x (naive wins) | 4.5 | 0.5 | 9x |
| 1024 | 1.088 | 0.881 | 1.24x | 17.0 | 1.0 | 17x |
| 2048 | 3.076 | 1.235 | **2.49x** | 66.0 | 2.0 | 33x |
| 4096 | 11.857 | **2.477** | **4.79x** | 260.0 | 4.0 | **65x** |

GFLOP/s (같은 연산량 4N²d + 3N² 기준):
- naive : 169 → 250 → 353 → 366
- flash : 115 → 308 → 879 → **1754** (T4 FP32 peak 8100 의 약 22%)

정확도: 두 구현 모두 max_abs_err < 5e-7 (double CPU 레퍼런스 대비 FP32 rounding 한계 내).

## 4. 네 가지 교훈

### (a) FA 의 **HBM 절감은 N² → N·d 스케일링**

N=4096, d=64 에서:
- naive HBM ≈ 4N² + 4Nd = 260 MB
- flash HBM ≈ 4Nd = 4 MB
- **비율 65x**, N 제곱으로 벌어짐

이게 FA 의 전부. 나머지는 이 절감을 실제 시간으로 환산하는 엔지니어링.

### (b) **Crossover**: N=512 에선 naive 가 더 빠름

이유:
- naive 의 Q@K^T, softmax, P@V 는 T4 L2 (4MB) 에 S (1MB) 가 들어감 → 실제 HBM 로드는 이론치보다 훨씬 적음
- flash 는 Q, K 를 여러 K/V 블록마다 재사용하지만 그래도 **recompute 오버헤드**가 있음 (Q 는 단일 로드라 괜찮지만 K/V 는 각 Q 블록마다 순회)
- Br=64 블록 수가 적은 N 에선 SM (T4=40개) 을 다 못 채움 → occupancy 낮음

**교훈**: "FA 가 항상 더 빠르다" 는 거짓. **sequence length 가 길어야** 우위가 나타남. 실제 LLM prefill (N=4096~32k) 에서 FA 가 결정적인 이유.

### (c) Flash 는 **HBM 절감 대비 실시간 속도 향상이 작다** (약 5x vs 65x HBM)

N=4096 에서 HBM 은 65x 줄어드는데 time 은 4.79x 만 빨라짐.

왜?
- naive 가 실제로는 HBM-bound 가 아님 (effective 22 GB/s → T4 의 320 GB/s 의 7%). **L2 가 대부분 흡수**.
- FLOPs 는 두 구현 동일 — flash 는 연산을 줄이는 게 아니라 **데이터 이동**만 줄임
- T4 의 FP32 peak 가 8 TFLOPS 인데 flash 는 1.75 TFLOPS 에서 멈춤 → **compute-bound 단계로 진입**
- FA 의 큰 성능은 **H100 같은 고대역 + 고peak** 에서 빛남 (FA-2/FA-3 의 논문 차트가 주로 A100/H100 인 이유)

T4 에서도 절감은 실재하지만 극적이지 않음.

### (d) Flash v1 = 레슨 4 + 레슨 5 의 **산술적 결합**

코드 줄바 한번 보면 투명:
- `S = Q @ K^T` 타일 연산 = 레슨 4 v3 의 register blocking 과 같은 구조 (여기선 thread-per-row 로 단순화)
- `alpha = __expf(m_i - m_new); l_new = alpha * l_i + l_tile` = **레슨 5 v3 의 online update 공식 그대로**
- `O_new = alpha * O + P @ V` = 누적된 출력을 새 max 로 재정규화 + 다음 타일의 contribution 가산

**5 개 레슨이 전부 이 80 줄에 수렴**. 이게 FA 논문이 "별로 새롭지 않은 기술의 날카로운 조합" 이라고 평가되는 이유.

## 5. 구조적 매핑

```
레슨 01 (vector_add)  →  coalesced load 패턴 (FA 의 Q/K/V 로드)
레슨 02 (memory)      →  HBM↔L2 트래픽 의식 (FA 의 존재 이유 자체)
레슨 03 (reduction)   →  warp reduce (naive softmax 의 row max/sum)
레슨 04 (matmul)      →  tiled matmul (FA 의 S = Q@K^T, O += P@V)
레슨 05 (softmax)     →  online (max, sum) update (FA 의 심장)
레슨 06 (flash attn)  →  위 5개의 fusion
```

## 6. LLM serving 번역

**Prefill phase** (긴 prompt 처리): N 수천~수만 → 우리 그림의 오른쪽 끝. FA 없이는 메모리도 모자람. vLLM/TensorRT-LLM 가 FA-2/FA-3 을 호출하는 지점.

**Decode phase** (토큰 1 개씩 생성): seq_len = 1 (현재 쿼리). Q 가 아주 작아서 FA 타일 구조 자체가 overkill. **Paged Attention** 등 다른 최적화가 중요.

우리의 단일 커널 FA 구현은 prefill 쪽의 dynamics 를 설명함.

## 7. 코드 읽기 지도 (유저가 복습 때 열어볼 순서)

1. [src/flash_attention.cu:64-100](/Users/xavier/dev/cudatraining/src/flash_attention.cu:64) — naive Q@K^T (타일 로드 패턴 복습)
2. [src/flash_attention.cu:104-170](/Users/xavier/dev/cudatraining/src/flash_attention.cu:104) — naive softmax (레슨 5 v2 와 같은 구조)
3. [src/flash_attention.cu:172-205](/Users/xavier/dev/cudatraining/src/flash_attention.cu:172) — naive P@V
4. [src/flash_attention.cu:222-320](/Users/xavier/dev/cudatraining/src/flash_attention.cu:222) — **flash_attention_v1 커널 본체**. 이 한 함수가 capstone.
5. [src/softmax.cu:200-270](/Users/xavier/dev/cudatraining/src/softmax.cu:200) (참고) — online softmax 의 원본

## 8. 다음 단계 후보

### Next A: 여기서 멈추고 9 시간 분산 코드 복습

유저가 예고한 경로. CUDA 레슨 1~6 의 커널을 각자 한 줄씩 읽고 해석. 핵심 단계: 레슨 4 v3 (register blocking) 과 본 레슨 v1 의 FA 커널은 특별히 정밀 리딩 가치 있음.

### Next B: Triton 포팅 (Phase 1 M1-2)

지금까지 6 개 커널 (vector_add / reduction / matmul / softmax / flash_attention) 을 Triton 으로 옮기고 "추상화의 비용" 블로그. 환경 세팅: Python, PyTorch, Triton. 본격 프로젝트 모드.

### Next C: FA 의 확장

- **Causal mask** (autoregressive attention) 추가 — 현재 masking 은 없음
- **FP16 + Tensor Cores** — register blocking 을 WMMA 로 전환 (레슨 4 v4 의 확장). T4 에서 2-3x 추가 기대
- **Batch + multi-head** — grid 에 (batch, head, q_block) 추가

FA-2 의 코어 idea 중 일부 (inner/outer 루프 swap, warp-level parallelism) 는 여기서 연습 가능.

### 추천

**A → B 순**. 레슨을 더 찍기보단 복습으로 뼈대 굳히고 Triton 으로 넘어가는 게 Phase 1 로드맵에 정확히 맞음. FA 확장 (C) 은 여력 있을 때.

## 9. VM 상태

`cuda-t4-dev-lesson02`: RUNNING. 유지.

## 10. 한 줄 요약

Flash Attention v1 을 **우리의 5 개 레슨 산출물을 결합**해 구현. **N=4096 에서 4.79x speedup, 65x HBM 절감** 관측. **Crossover 는 N≈512-1024** — FA 는 긴 context 전용의 무기라는 **경계선**을 숫자로 확인. CUDA 의 "레벨 1" 학습 여기서 종료, 이후는 복습 + Triton + 프로젝트 모드.
