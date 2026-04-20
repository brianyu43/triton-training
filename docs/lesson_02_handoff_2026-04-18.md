# Lesson 02 Handoff

기준 날짜: `2026-04-18`

주제: **복사 비용의 정체 — pageable vs pinned host memory**

## 1. 이번 세션에서 한 일

### 코드 변경

- [src/vector_add.cu](/Users/xavier/dev/cudatraining/src/vector_add.cu:1) 에 `--pageable` / `--pinned` 플래그 추가
- CSV 출력에 `mode` 컬럼 추가
- host 버퍼 할당을 `cudaMallocHost` (pinned) 와 `new float[]` (pageable) 로 분기

### 자동화

- [scripts/run_pinned_vs_pageable.sh](/Users/xavier/dev/cudatraining/scripts/run_pinned_vs_pageable.sh:1) — 16 runs 스윕 (`2^14 .. 2^28` × `{pinned, pageable}`)
- [scripts/gcp_run_lesson02.sh](/Users/xavier/dev/cudatraining/scripts/gcp_run_lesson02.sh:1) — GCP 전체 파이프라인 (copy → build → sweep → pull CSV)
- [scripts/gcp_create_t4_spot_vm.sh](/Users/xavier/dev/cudatraining/scripts/gcp_create_t4_spot_vm.sh:1) 에 `--no-service-account --no-scopes` 추가 (기본 SA 없는 프로젝트 대응)

### 실행

- VM: `cuda-t4-dev-lesson02` (T4 spot, `us-east1-d`, 유지)
- 16 runs 완료, `best_ms` / `avg_ms` / `h2d_ms` / `d2h_ms` CSV로 회수

## 2. 산출물

- [results/remote/pinned_vs_pageable_t4.csv](/Users/xavier/dev/cudatraining/results/remote/pinned_vs_pageable_t4.csv:1) — 16 rows
- [results/lesson02-run-20260418-145713.log](/Users/xavier/dev/cudatraining/results/lesson02-run-20260418-145713.log:1) — 전체 실행 로그

## 3. 가공된 핵심 숫자

### Transfer bandwidth plateau (큰 n 기준)

| | pinned | pageable | ratio |
|---|---:|---:|---:|
| H2D | ~12.3 GB/s | ~4.6 GB/s | 2.7x |
| D2H | ~13.1 GB/s | ~1.03 GB/s | **12.7x** |

### End-to-end total (H2D + best kernel + D2H)

| n | pinned | pageable | ratio |
|---|---:|---:|---:|
| 2^20 | 1.12 ms | 6.84 ms | 6.1x |
| 2^22 | 4.28 ms | 24.09 ms | 5.6x |
| 2^24 | 16.87 ms | 95.03 ms | 5.6x |
| 2^26 | 67.38 ms | 375.69 ms | 5.6x |
| 2^28 | 270.56 ms | 1519.26 ms | 5.6x |

커널 시간은 두 모드에서 동일 (동일 GPU 경로). 차이는 **100% host-side 복사**에서 나온다.

## 4. 해석 — 이번 레슨에서 건진 3가지

### (a) pageable 페널티는 방향별로 크게 다르다

- H2D pageable: 2.7x 느림 (이론 PCIe 대비 ~40%)
- D2H pageable: 12.7x 느림 (이론 대비 ~8%)

차이의 원인은 **write target 페이지의 상태**다. `new float[n]` 은 가상 주소만 할당하고 물리 페이지를 커밋하지 않는다. D2H에서 드라이버가 staging → pageable로 CPU `memcpy` 할 때 매 페이지에 **demand-zero fault**가 발생. 1 GB 버퍼면 ~262k 페이지 fault.

H2D는 반대 방향 — 유저가 init 루프에서 이미 touch 한 페이지를 읽기만 한다. Fault 없음.

**실전 규칙**: pageable이 어쩔 수 없이 강제될 때도, **output buffer는 미리 touch 해두라** (`memset` 한 번, 또는 첫 iteration을 throwaway로). 이것만으로도 D2H가 수 배 빨라진다.

### (b) T4는 pinned일 때 PCIe Gen3 실효 상한에 닿는다

- H2D 12.3 / D2H 13.1 GB/s
- PCIe Gen3 x16 이론 16 GB/s, 실효 ~13
- 즉 **더 짜낼 여지 없음**. 다음 레버는:
    - **겹치기** (streams + async copy로 kernel과 transfer overlap)
    - **제거** (persistent device buffers, unified memory, zero-copy)
    - **감량** (kernel fusion으로 round-trip 줄이기)

이게 왜 vLLM이 KV cache를 GPU에 붙박이로 두는지, 왜 operator fusion이 이득이 큰지의 배경이다.

### (c) 작은 n은 다른 세계다

- n ≤ 2^18: 커널이 launch overhead floor (~4-7 μs)에 눌림. "bandwidth %" 개념 자체가 의미 없음.
- n = 2^18 에서 apparent bandwidth가 이론치의 **140%** — L2 cache hit에 의한 허위 대역폭.
- Crossover to HBM-bound: **n ≈ 2^20 (4 MB)**.

서빙 관점에서: 작은 배치 / 작은 시퀀스는 kernel 자체 최적화보다 **batching / launch 감소** 가 먼저. CUDA Graphs 가 이 구간에서 큰 차이를 만든다.

## 5. 구조적 레슨

이번 주제의 핵심 문장:

> Memory-bound 워크로드에서 **가장 큰 레버는 kernel이 아니라 data path 다.**
> 커널이 이론 대역폭의 70%를 찍고 있어도, host 쪽을 pageable로 두면 end-to-end는 5배 느려진다.

지난 Lesson 01이 `연산보다 이동이 비싸다`였다면, Lesson 02는 `이동 중에서도 host memory 선택이 가장 크다`다.

## 6. 다음 레슨 후보

### Next A: CUDA streams + async copy overlap

- 현재는 H2D → kernel → D2H가 직렬
- 스트림으로 H2D 와 kernel 을 겹치면 전체 latency 얼마나 줄어드나?
- 큰 n은 kernel(15ms) vs H2D(170ms) 비대칭이라 overlap 이득이 제한적일 것 — 실제로 측정해봐야.
- 새 flag: `--stream-overlap`, 벤치마크는 chunked copy/compute

### Next B: reduction 커널로 넘어가기

- vector add 는 "한 번 읽고 한 번 쓰기" → 극단적으로 단순
- reduction은 "병렬 + shared memory + warp shuffle" 을 처음 만나는 지점
- `sum(x)` 구현: naive → shared memory → warp shuffle → 최종 cudaDevice primitive 비교
- 포트폴리오 관점에서 이쪽이 M1-2 단계에 맞는다 ([project_phase1_plan.md](/Users/xavier/.claude/projects/-Users-xavier-dev-cudatraining/memory/project_phase1_plan.md) 기준)

### Next C: block size / grid size 재검토 (소규모)

- 지금 block_size=256 고정, grid_size는 `min(ceil(n/bs), SM*32)`
- n이 작을 때 grid_size가 작아 SM 미포화. 이게 small-n floor의 일부인지 측정
- 30분짜리 서브레슨

**추천**: **Next B (reduction)** 가 Phase 1 로드맵에 정렬된다. Streams overlap은 Month 2-3에 reduction/matmul 후 붙이는 게 자연스러움.

## 7. 다음 레슨 체크리스트 (Next B 선택 시)

1. `src/reduction.cu` 새 파일 — naive global-memory reduction 부터
2. 버전 4개 구현: 
   - v1 naive (atomicAdd per thread)
   - v2 shared memory tree reduction
   - v3 sequential addressing + unrolled last warp
   - v4 warp shuffle (`__shfl_down_sync`)
3. `--version {1,2,3,4}` 플래그로 비교
4. cuBLAS `cublasSasum` 또는 thrust `thrust::reduce` 와 맞비교
5. `results/reduction_t4.csv` — 같은 스윕 포맷

## 8. VM 상태

- `cuda-t4-dev-lesson02` : **RUNNING**, 유지
- 다음 세션에 그대로 재사용. 시작 시 `gcloud compute ssh` 로 접속만 하면 됨. 
  (주의: spot VM이므로 GCP가 임의 종료 가능. 다음 세션 시작 시 `gcloud compute instances describe` 로 상태 먼저 확인.)

## 9. 클로드에게 넘길 프롬프트 초안 (Next B용)

```text
/Users/xavier/dev/cudatraining/docs/lesson_02_handoff_2026-04-18.md 를 읽고 Next B (reduction) 로 이어가자.

현재 상태:
- T4 VM cuda-t4-dev-lesson02 는 us-east1-d 에서 유지 중 (바로 ssh 가능)
- vector_add.cu 는 pinned/pageable 모두 지원, sweep 결과 정리 완료

원하는 것:
- src/reduction.cu 작성 — v1 (atomic) / v2 (shared tree) / v3 (unrolled) / v4 (warp shuffle)
- 각 버전마다 best_ms, effective_gbps, SM occupancy 측정
- cuBLAS / thrust baseline 비교 추가
- 결과를 results/reduction_t4.csv 에 저장하고 해석까지 정리해줘
```

## 10. 한 줄 요약

Lesson 02 성공. `복사 비용 5.6배`와 `D2H pageable 페이지 fault`를 숫자로 잡았다. PCIe 상한에 닿는 감각까지 확보. 다음은 reduction 으로 **shared memory / warp primitives** 첫 접촉.
