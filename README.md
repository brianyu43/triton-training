# triton-training

CUDA → Triton 커널을 lesson 단위로 쌓는 개인 훈련 레포. 각 커널은 *측정과 해석이 가능한 형태* 로 구현하고, lesson별 handoff 문서에 의사결정/속도/막힌 포인트를 남긴다.

6개월 Phase 1 목표: vLLM / Triton / Flash Attention 같은 GPU 서빙 스택에서 판단 가능한 엔지니어로 착지.
-> 손코딩열심히 
---

## 현재 범위 (Lesson 01 → 12)

| #  | 커널 / 세션                  | CUDA | Triton    | 핵심 포인트 |
|----|------------------------------|------|-----------|------------|
| 01 | Vector Add                   | ✅   | ✅ (smoke) | 메모리 대역폭 기준선, launch config |
| 02 | Reduction                    | ✅   | ✅        | warp shuffle, 블록 리덕션, atomic 대안 |
| 03 | Softmax                      | ✅   | ✅        | numerically-stable, row-wise |
| 04 | Pinned vs Pageable           | ✅   | —         | H2D/D2H 전송 오버헤드 |
| 05 | Matmul                       | ✅   | ✅        | tiled → block-tiled → warp-tiled |
| 06 | Matmul 확장                  | ✅   | —         | shared memory padding, double buffer |
| 07 | Flash Attention v1           | ✅   | —         | online softmax, 단일 타일 |
| 08 | Flash Attention (Triton)     | —    | ✅        | autotune, fp16/fp32, N=8192에서 우리 CUDA 대비 **6.14×** |
| 09 | MHA + Causal Flash Attention | —    | ✅        | `(B, H, N, d)` 4-D, `IS_CAUSAL` constexpr, 루프 스킵, `torch.ops` 등록 |
| 10 | Profiling (nsys + ncu)       | —    | —         | 레슨 1-9 재해석: pinned D2H 10×, reduction stall = `lg_throttle`, lesson 09 의 20% gap = register pressure |
| 11 | Paged Attention (vLLM-style) | —    | ✅        | block_table indirection, GQA grid refactor `(B, H_q)→(B, H_kv)`, LLaMA-3-8B 에서 SDPA **-14 %** (우리가 빠름), vLLM Triton unified 와 axis-for-axis 매치 |
| 12 | Paged Attention Split-K      | —    | ✅        | vLLM v2-style `(B, H_kv, SEGMENTS)` + reduce kernel, auto-dispatch heuristic, **MQA paged 1.68× 가속** (SM 점유율 구멍 닫음) |

각 lesson handoff 문서: [`docs/lesson_XX_handoff_*.md`](./docs/).
블로그 초안 (Triton vs CUDA): [`docs/blog_draft_triton_vs_cuda.md`](./docs/blog_draft_triton_vs_cuda.md).

---

## 헤드라인 결과 (L4 GPU, sm_89)

- **Triton fp16 matmul**: ~54 TFLOPS (cuBLAS 대비 ~77%)
- **Triton fp16 Flash Attention**, N=8192, d=64: lesson 07의 우리 CUDA FA v1 대비 **6.14× 빠름**
- **Triton fp16 Paged Attention** (LLaMA-3-8B GQA, B=8, ctx=2k): cuDNN / FA-2 SDPA 대비 **-14 % (우리가 빠름)**. 275줄 Triton 커널이 production CUDA 를 이긴 shape.
- **Triton fp16 Paged Attention Split-K** (MQA, B=16, H_kv=1, ctx=4k): single-pass 대비 **1.68× 가속** (0.331 → 0.197 ms) — vLLM v2 의 `(B, H_kv, SEGMENTS)` + reduce kernel 구조로 SM 점유율 28 % → 100 % 회복.
- 원인 분해 — autotune (`BLOCK_M/N`, `num_warps`, `num_stages`) + 온라인 softmax + `tl.dot` HW matrix path + paged 의 `(B, H_kv)` grid refactor + split-k 의 ctx-축 추가 병렬화

> 이 숫자는 L4 (Ada Lovelace, sm_89) 한 장에서만 유효. 다른 GPU에서는 autotune을 다시 타야 의미 있음.

---

## Lesson 09 — 4-D MHA + Causal Flash Attention

- **인터페이스**: `(B, H, N, d)` — 실제 LLM forward shape
- **3-D 그리드**: `(cdiv(N, BLOCK_M), H, B)` → 각 program이 하나의 `(batch, head)` Q 블록 담당
- **Causal mask**: `IS_CAUSAL: tl.constexpr` → Triton이 causal/non-causal 두 커널로 특수화 컴파일, 런타임 브랜치 비용 0
- **FA-v2 루프 스킵**: causal일 때 상삼각 타일을 아예 건너뜀 — `end_n = min(N, (pid_m+1)*BLOCK_M)` → 평균 ~50% K/V 스캔 단축

진행 상태:
- ✅ Phase 1: 2-D → 4-D 확장, non-causal 정확도 통과 (57 shape)
- ✅ Phase 2: causal mask + 루프 스킵, 정확도 통과 (60 shape, `N=129/513` edge case 포함)
- ✅ Phase 3: 3-way speed bench (ours / SDPA / naive) on LLaMA-7B + GPT-2 형태
- ✅ Phase 4: `torch.library.custom_op` 등록, `torch.compile(fullgraph=True)` 통과
- ✅ Phase 5: [lesson 09 handoff](./docs/lesson_09_handoff_2026-04-20.md) + [블로그 초안](./docs/blog_draft_lesson_09_mha_causal_fa.md)

### Phase 3 headline (L4, fp16, median of 100 reps)

| Shape (LLaMA-7B causal, d=128)   | ours        | SDPA (FA-2) | ours/SDPA | vs naive |
|----------------------------------|-------------|-------------|-----------|----------|
| B=1 H=32 N=1024                  | 0.223ms (38.5 TF) | 0.202ms | 0.90× | 30× |
| B=1 H=32 N=2048                  | 0.784ms (43.8 TF) | 0.613ms | 0.78× | 31× |
| B=1 H=32 N=4096                  | 2.964ms (46.4 TF) | 2.559ms | 0.86× | 33× |
| B=4 H=32 N=1024                  | 0.886ms (38.8 TF) | 0.726ms | 0.82× | 29× |

| Shape (GPT-2 causal, d=64)       | ours              | SDPA    | ours/SDPA |
|----------------------------------|-------------------|---------|-----------|
| B=8  H=12 N=1024                 | 0.302ms (42.7 TF) | 0.303ms | **1.00× 동률** |
| B=16 H=12 N=512                  | 0.249ms (25.9 TF) | 0.282ms | **1.13× 우세** |

요지: ~155줄 Triton이 d=128 LLaMA-7B 형태에서 FlashAttention-2 대비 78–90% 속도. d=64에서는 동률 또는 우세. naïve 대비 29–33×.

---

## Lesson 10 — nsys + ncu 로 레슨 1-9 뜯어보기

새 커널을 더 짜는 대신, 이미 짠 9 개 커널을 **Nsight Systems / Nsight Compute 로 프로파일해서 "왜 그 숫자인지" 를 숫자로 고정** 하는 세션. 4 Phase.

| Phase | 대상 | 도구 | 핵심 발견 |
|---|---|---|---|
| 1 | lesson 04 pinned vs pageable | `nsys` | **pinned D2H 는 pageable D2H 보다 9.9× 빠름** — pageable D2H 가 2-hop memcpy 라 H2D 의 1/3.6 속도. 커널 시간은 고정 (0.834 ms) |
| 2 | lesson 02 reduction v1 vs v4 | `ncu` | v1 과 v4 의 **occupancy 가 같다 (91 %)**. 218× 차이는 v1 의 `lg_throttle` 31 % (atomic L2-line serialization). DRAM 은 0.46 % 로 비어 있고 L2 hit 이 88.7 % 로 hot-line 을 전 SM 이 다툼 |
| 3 | lesson 09 ours vs SDPA (FA-2) | `ncu` | SDPA backend 는 cuDNN 이 아니라 **Tri Dao FA-2 CUDA 구현**. 22 % gap = register pressure (**255 regs/thread → occupancy 반토막**) + MMA `wait` 38.6 %. SDPA 는 이미 `math_pipe_throttle` 41 % (healthy compute-bound) |
| 4 | 핸드오프 + 블로그 | — | [`docs/lesson_10_handoff_2026-04-20.md`](./docs/lesson_10_handoff_2026-04-20.md), [`docs/blog_draft_lesson_10_profiling.md`](./docs/blog_draft_lesson_10_profiling.md) |

세 줄 요약:
- **Occupancy 는 throughput 이 아니다** — Phase 2 와 Phase 3 둘 다 "occupancy 는 같거나 가까운데 throughput 이 달라" 로 증명.
- **Stall reason 분포 = 커널 지문** — `long_scoreboard` (memory), `math_pipe_throttle` (healthy compute), `wait` (MMA dep), `lg_throttle` (atomic) 중 dominant 한 게 뭐냐로 처방이 달라진다.
- **큰 tile 이 빠를 거라는 직관은 틀릴 수 있다** — 우리 autotune 이 `BLOCK_M=128` 을 골랐지만 register 255 로 spill 직전, occupancy 절반. SDPA 의 `64` tile 이 그 shape 에서는 더 빠름.

결과물: `results/lesson10_phase{1,2,3}/summary.md` + `.nsys-rep` / `.ncu-rep` 파일 (Nsight GUI 로 열람 가능).

---

## Lesson 11 — Paged Attention (vLLM-style) in Triton

Lesson 09 의 contiguous 4-D MHA 를 **vLLM 의 paged KV cache** 구조로 재작성. block 단위로 쪼갠 KV pool + sequence 별 `block_table` 로 간접 참조. 6 Phase (0 = reference, 1 = MHA decode, 2 = GQA/MQA, 3 = speed bench, **3.5 = grid refactor + IEEE fix**, 4 = vLLM 소스 diff).

### 커널 구조

```
Q            : (B, H_q,  d)                     ← decode 니까 N=1
K_cache      : (num_blocks, block_size, H_kv, d) ← 고정 크기 블록 풀
V_cache      : same
block_table  : (B, max_blocks_per_seq) int32    ← logical → physical
context_lens : (B,) int32

grid = (B, H_kv)     ← Phase 3.5: one program per (batch, kv_head)
                       program 안에서 GQA_GROUP_SIZE query heads 묶어 처리
```

### Phase 3.5 — Grid refactor + silent TF32 fix

Phase 3 의 속도 벤치에서 **GQA gap 이 `GROUP_SIZE` 에 비례 (+16 % ~ +1316 %)** 라는 구조적 패턴 발견. grid `(B, H_q)` 는 group 내 query heads 가 독립적으로 KV 를 중복 로드 → DRAM 낭비.

수정: grid → `(B, H_kv)`, program 안에서 Q 를 `(GROUP, HEAD_DIM)` 2D tile 로 로드, `tl.dot` (GROUP≥4) 로 group 전체 score 계산. K/V block 은 program 당 한 번만 로드.

동시에 `tl.dot(fp32, fp32)` 가 sm_80+ 에서 **기본 TF32** (10-bit mantissa) 로 떨어지는 함정 포착 → `input_precision="ieee"` 로 강제, fp32 max diff 4.1e-04 → 3.6e-07 복구.

### Phase 3.5 headline (L4, fp16, warmup=50, iters=200)

| Shape | B | H_q | H_kv | group | SDPA ms | paged best (bs) | gap |
|---|---|---|---|---|---|---|---|
| llama7b MHA ctx=2k | 8 | 32 | 32 | 1 | 1.322 | 1.227 (bs=16) | **-7 %** |
| llama7b MHA ctx=8k | 8 | 32 | 32 | 1 | 6.115 | 4.927 (bs=64) | **-19 %** |
| **llama38b GQA ctx=2k** | **8** | **32** | **8** | **4** | **0.308** | **0.264 (bs=16)** | **-14 %** ← SDPA 이김 |
| llama38b GQA ctx=2k | 32 | 32 | 8 | 4 | 1.163 | 1.197 (bs=128) | +3 % |
| llama70b GQA ctx=2k | 4 | 64 | 8 | 8 | 0.049 | 0.048 (bs=128) | **-1 %** |
| llama70b GQA ctx=4k | 8 | 64 | 8 | 8 | 0.532 | 0.526 (bs=16) | **-1 %** |
| mqa ctx=4k | 16 | 32 | 1 | 32 | 0.048 | 0.089 (bs=128) | +85 % (L2 reuse 한계) |

Correctness: **32/32 PASS** (fp16 max diff 9.8e-04, fp32 3.6e-07).

### Phase 3 → Phase 3.5: GQA gap 개선폭

| shape | Phase 3 gap | Phase 3.5 gap | Δ |
|---|---|---|---|
| llama38b group=4 B=8 | +161 % | **-14 %** | 226 % 점 |
| llama38b group=4 B=32 | +86 % | +3 % | 83 % 점 |
| llama70b group=8 | -1 % | -1 % | 이미 이겼음 |
| mqa group=32 | +1316 % | **+85 %** | 1231 % 점 |

### Phase 4 — vLLM 소스 읽고 diff

`vllm/v1/attention/ops/triton_unified_attention.py` 가 **내 Phase 3.5 와 axis-for-axis 매치** — 같은 `(..., H_kv)` grid, 같은 `(GROUP, HEAD)` Q tile, 같은 KV layout `(num_blks, blk_size, H_kv, d)`, 같은 `tl.dot(Q, K)` / `tl.dot(P, V)` 패턴. vLLM 자신이 2023 CUDA v1 의 per-query-head grid 에서 per-KV-head 로 refactor 한 역사 — **GQA shapes 가 널리 쓰이면서 같은 forcing function 이 발생했음**. 내가 독립적으로 수렴한 게 설계 증거.

Split-k (ctx 축 partition + reduce) 는 vLLM v2 / `kernel_unified_attention_3d` 에 있음. MQA 잔여 +85 % 를 닫는 해법으로 Lesson 12 입구에 blueprint 로 이월.

진행 상태:
- ✅ Phase 0: PyTorch reference oracle
- ✅ Phase 1: Triton MHA decode kernel
- ✅ Phase 2: GQA + MQA 일반화 (32/32 correctness)
- ✅ Phase 3: 속도 벤치 + GQA 구조 gap 관측
- ✅ Phase 3.5: grid refactor + IEEE fix (LLaMA-3-8B 에서 SDPA 이김)
- ✅ Phase 4: vLLM 소스 diff, axis-for-axis 매치 확인
- ✅ Phase 5: [lesson 11 handoff](./docs/lesson_11_handoff_2026-04-22.md) + [블로그 초안](./docs/blog_draft_lesson_11_paged_attention.md)

세 줄 요약:
- **Correctness 가 통과해도 구조 버그는 속도로만 보인다** — Phase 2 의 32/32 PASS 직후 "끝" 이라 판단했으면 GQA shape 에서 2-13× 느린 커널을 shipping 했을 것.
- **버그가 버그 뒤에 숨는다** — Grid bug (Phase 3 이 드러냄) 를 고치자마자 TF32 bug (Phase 3.5 에서 드러남) 가 튀어나옴. 큰 refactor 뒤엔 correctness 재실행 필수.
- **독립적으로 수렴하는 게 설계가 맞다는 증거** — vLLM 소스를 Phase 3.5 끝난 뒤 읽었는데 현행 Triton 포트와 axis-for-axis 매치. 내가 똑똑한 게 아니라 맞는 답이 하나.

결과물: `triton_kernels/paged_attention.py` (275 lines), `docs/lesson_11_phase{3_findings,4_vllm_diff}.md`, `results/lesson11-phase{1,3,3.5}-*.log`.

---

## Lesson 12 — Paged Attention Split-K (vLLM v2-style)

Lesson 11 Phase 4 에서 관찰된 MQA 의 SM 점유율 구멍 (B\*H_kv=16 에 SM=58, **28 % 점유**) 을 vLLM 의 `paged_attention_v2.cu` 와 동일한 **split-k + reduce 2-kernel 구조** 로 해결. 4 Phase.

### 커널 구조 확장

```
기존 (Lesson 11):
  paged_attention_decode_kernel   grid = (B, H_kv)          ← single-pass
  → each program walks ALL ctx for its (batch, kv_head)

추가 (Lesson 12):
  paged_attention_split_kernel    grid = (B, H_kv, SEGMENTS) ← PARTITION_SIZE=512 토큰씩
  → writes UNNORMALIZED (m_i, l_i, acc) to scratch
  paged_attention_reduce_kernel   grid = (B, H_q)            ← SEGMENTS 축 재결합
  → alpha = exp(m_s - m_global), 표준 online softmax

Wrapper 의 auto-dispatch:
  use_split_k = (B*H_kv < 0.5*SM_COUNT) AND (segments >= 4)
```

### Phase 3 path comparison (L4, fp16, bs=16, partition_size=512)

| shape | SDPA ms | SP ms | SK ms | auto ms | auto 선택 | SK vs SP |
|---|---|---|---|---|---|---|
| llama7b-B1-ctx1k   | 0.034 | 0.142 | 0.196 | 0.143 | **SP** | -38 % |
| llama7b-B1-ctx4k   | 0.264 | 0.475 | 0.389 | 0.466 | SP (miss) | +18 % |
| llama7b-B8-ctx2k   | 1.184 | 1.140 | 1.206 | 1.144 | **SP** | -6 % |
| llama38b-B8-ctx2k  | 0.269 | 0.326 | 0.393 | 0.327 | **SP** | -21 % |
| llama70b-B4-ctx2k  | 0.043 | 0.165 | 0.196 | 0.166 | **SP** | -19 % |
| **mqa-B16-ctx4k**  | **0.044** | **0.331** | **0.196** | **0.197** | **SK** | **+41 %** |

**MQA 에서 paged 자체 1.68× 가속** (0.331 → 0.197 ms), 10 shape 중 9 에서 auto-dispatch 가 빠른 경로를 고름 (miss 1 건은 17 % gain 놓침 — regression 안 내는 쪽으로 보수적 튜닝).

### 진행 상태
- ✅ Phase 1: split_kernel + reduce_kernel + auto-dispatch wrapper
- ✅ Phase 2: correctness 32/32 PASS (single-pass + split-k 양쪽), fp16 max diff 9.8e-04
- ✅ Phase 3: speed bench + heuristic 튜닝 (0.75·SM → 0.5·SM, segments≥2 → ≥4)
- ✅ Phase 4: [lesson 12 handoff](./docs/lesson_12_handoff_2026-04-22.md) + [블로그 초안](./docs/blog_draft_lesson_12_split_k.md)

### 세 줄 요약
- **파라미터 스윕이 반응하지 않으면 아키텍처를 의심해라** — Lesson 11 에서 block_size 8~128 어느 것을 써도 MQA 가 안 닫혔음. 그 시점이 architecture 를 갈아야 한다는 신호였다.
- **Sentinel 값이 무시 동작과 수학적으로 동등한 자리는 코드를 단순하게 만든다** — Invalid segment 와 pow-of-2 padding lane 둘 다 `m=-inf` 초기값으로 통일, reduce 의 `exp(-inf - m_global)=0` 로 자동 캔슬.
- **Auto-dispatch 는 regression 안 내는 게 첫 번째, 이득은 두 번째** — 첫 heuristic (0.75·SM) 이 LLaMA-70B 에서 SK 를 잘못 골라 20 % 느려짐. 0.5·SM 으로 조이니 9/10 correct, 1 miss 는 17 % gain 만 놓침 (수용).

### SDPA gap 이 여전한 이유
MQA 에서 SDPA 대비 gap 은 여전히 +344 %. 원인은 SM 점유율이 아니라 **L2 residency**. K/V 총량 16 MB 가 L4 의 48 MB L2 에 완전히 들어가서 SDPA 는 실제로 **761 GB/s 등가** (L4 DRAM peak 300 GB/s 초과) — L2 에서 읽고 있음. Paged 는 `block_table` 간접참조 + block 단위 gather 로 L2 spatial locality 가 깨져서 DRAM bound (165 GB/s). 이 gap 은 split-k 로 닫을 수 없는 영역 — 다음 세션 거리로 남김.

결과물: `triton_kernels/paged_attention.py` (~550 lines, 기존 + split + reduce), `docs/lesson_12_handoff_2026-04-22.md`, `docs/blog_draft_lesson_12_split_k.md`, `results/lesson12-{correctness,speed}-*.log`.

---

## 빌드 & 실행

### CUDA (lessons 01–07)
```bash
make vector_add
./bin/vector_add --n 67108864 --block-size 256 --iterations 100
```

다른 커널도 같은 패턴: `make reduction`, `make softmax`, `make matmul`, `make flash_attention`.

### Triton (lessons 05, 07, 08, 09, 11)
```bash
# Lesson 09 · correctness (non-causal + causal)
python3 triton_kernels/bench/bench_flash_attention_mha.py

# Lesson 09 · 3-way speed bench (ours vs SDPA vs naive)
python3 triton_kernels/bench/bench_flash_attention_mha_speed.py

# Lesson 11+12 · paged attention correctness (32 shapes × 2 dtypes, single-pass + split-k 양쪽)
python3 triton_kernels/bench/bench_paged_attention.py

# Lesson 11 · speed bench (SDPA vs paged, 10 shapes × 5 block sizes)
python3 triton_kernels/bench/bench_paged_attention_speed.py

# Lesson 12 · path comparison (SP vs SK vs auto at block_size=16)
python3 triton_kernels/bench/bench_paged_attention_speed.py --compare-paths
```

### 클라우드 VM (GCP)

로컬 Mac (Apple Silicon)에는 NVIDIA GPU가 없어서, 실행은 GCP L4 spot (lessons 07–09) 또는 T4 spot (lessons 01–06)에서 한다:

```bash
# 1회: VM 생성 (G2 family = L4 1장)
./scripts/gcp_create_l4_spot_vm.sh <PROJECT_ID> <ZONE> <VM_NAME>

# lesson 실행마다: 레포 sync + run
./scripts/gcp_run_lesson09_phase2.sh <PROJECT_ID> <ZONE> <VM_NAME>
```

---

## 출력에서 볼 것 (lesson 01 스타일 기준)

프로그램은 아래를 출력한다.

- GPU 이름과 대략적인 theoretical memory bandwidth
- kernel best/avg time
- effective bandwidth
- H2D / D2H copy time
- CPU reference와의 최대 오차

처음에는 숫자 자체보다 아래 질문이 중요하다.

- block size를 바꾸면 왜 거의 안 변하거나, 어느 지점부터만 변하는가
- effective bandwidth가 theoretical bandwidth 대비 몇 퍼센트인가
- kernel time과 H2D/D2H 복사 시간 중 어디가 더 큰가

---

## 디렉토리 구조

```
src/                CUDA 커널 (.cu) — lessons 01–07
triton_kernels/     Triton 커널 (.py) + bench/ 서브디렉토리
docs/               lesson별 handoff 문서 + 블로그 초안
scripts/            GCP VM create/copy/run 스크립트
extension/          (legacy) PyTorch C++ extension — lesson 01 초기 실험
Makefile            CUDA 커널 빌드 타겟
```

---

## 의도

- **공개 artifact 중심** — commit / bench output / handoff 문서 자체가 포트폴리오.
- **학위가 아니라 측정 결과로** — cuBLAS 대비 %, torch SDPA 대비 × 단위로 줄을 긋는다.
- **독학 기록** — 막혔던 지점, 잘못 판단했던 포인트를 lesson handoff에 그대로 남긴다.

## 라이선스

학습용 개인 저장소.
