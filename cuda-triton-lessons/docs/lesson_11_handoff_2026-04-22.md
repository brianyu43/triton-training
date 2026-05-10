# Lesson 11 Handoff — Paged Attention (vLLM-style) in Triton

기준 날짜: `2026-04-22`

주제: **vLLM 의 핵심 커널 (paged attention decode) 을 Triton 으로 직접 재구현.** Lesson 09 의 contiguous `(B, H, N, d)` MHA 를 **block table indirection** 구조로 재작성. MHA / GQA / MQA 모두 동작, 실제 LLaMA-3-8B shape 에서 SDPA 보다 빠름.

세션 규모: 6 Phase (0 = Python reference, 1 = Triton decode kernel v1 MHA, 2 = GQA + MQA 일반화, 3 = 속도 벤치 + GQA gap 관측, **3.5 = grid refactor + IEEE fix**, 4 = vLLM 소스 diff, 5 = 이 문서).

---

## 1. 이번 세션에서 한 일

### 왜 이 세션이 필요한가

Lesson 09 에서 contiguous 4-D MHA 를 짰다. 그 커널은 이미 SDPA (=Tri Dao FA-2 CUDA) 의 78-90 % 속도. 그런데 **실전 LLM 서빙은 contiguous KV cache 를 쓰지 않는다**. vLLM 이 SOSP '23 에서 논문까지 내면서 밀어붙인 이유는 단순: contiguous cache 는 seq 길이가 가변이면 pre-allocate 가 낭비, GPU 메모리의 70 % 이상을 fragmentation 으로 날린다.

이 세션의 목적은 한 문장으로:

> Lesson 09 의 커널에 **block_table 경유 2-step 로딩** 만 끼워넣으면 vLLM 의 paged attention 이 되는지, 된다면 어떤 함정들이 있는지, 숫자로 고정한다.

실제로는 "block_table 만 끼워넣으면 된다" 는 순진한 예상이 **두 번** 틀렸다 (Phase 3 과 Phase 3.5). 그 두 번의 실패가 이 세션의 알맹이.

### Phase 0 — Python reference (2 시간)

- [triton_kernels/paged_attention_ref.py](/Users/xavier/dev/cudatraining/triton_kernels/paged_attention_ref.py:1) — PyTorch naive 구현 (loop 로 block_table 돌면서 gather, 그 뒤 표준 attention).
- 검증: 같은 contiguous KV 를 `pack_kv_paged()` 로 paged 로 포맷 변환 → `paged_ref_out == contiguous_naive_out` (rtol=1e-5, fp32) 통과.

paged 구조의 최소 케이스 (B=1, H=1, block_size=16, ctx=32, 2 blocks) 로 smoke 가 바로 돌고, 이 오라클을 이후 모든 Triton 커널의 correctness baseline 으로 썼다.

### Phase 1 — Triton decode kernel v1 (MHA 만) (1 일)

- [triton_kernels/paged_attention.py](/Users/xavier/dev/cudatraining/triton_kernels/paged_attention.py:1) — 메인 커널.
- 그리드: **초기에는 `(B, H_q)` — 하나의 program 이 하나의 `(batch, query head)` 를 담당**. Lesson 09 의 자연스러운 연장.
- 제약 단순화: MHA 만, fp16/fp32 둘 다, block_size ∈ {8, 16, 32, 64, 128}.
- Correctness: 10 shape × 2 dtype = 20/20 PASS, fp16 max diff 1e-3 이하, fp32 1e-5 이하.

이 시점의 함정 1 (기록):
- `decode 라 Q=1` 이지만 `tl.dot` 의 최소 M=16 제약. 초기 v1 은 **manual broadcast dot (`tl.sum(q*k)`)** 로 피해감. `tl.dot` 은 Phase 3.5 에서 도입.
- `tl.load(block_table, dtype=int32)` 을 잊으면 Triton 이 int64 로 해석 → pointer arithmetic 이 `* 8` 스케일 되면서 주소가 어긋남. Phase 0 oracle 이 없었으면 놓쳤을 버그.

### Phase 2 — GQA + MQA 일반화 (반나절)

- 커널에 `GQA_GROUP_SIZE: tl.constexpr` 추가. `kv_head = pid_h // GQA_GROUP_SIZE` 로 간접참조.
- 변경된 코드 라인: **총 4 줄** (constexpr 인자 1 + kv_head 계산 1 + K/V load base 의 `pid_h` → `kv_head` 2 곳).
- Correctness: **16 shape × 2 dtype = 32/32 PASS**:
  - LLaMA-3-8B (group=4): B=2, H=32, H_kv=8, d=128, fp16 max diff 2.44e-04
  - LLaMA-70B (group=8): B=4, H=64, H_kv=8, d=128, 가변 ctx, fp16 max diff 3.05e-05
  - MQA (group=16): B=2, H=16, H_kv=1, d=64, fp16 max diff 1.91e-06

이 시점까지는 **"쉽다, 잘 된다"** 느낌. Correctness 는 다 통과. 속도를 보자.

### Phase 3 — 속도 벤치, GQA 가 구조적으로 느리다는 발견 (반나절)

- [triton_kernels/bench/bench_paged_attention_speed.py](/Users/xavier/dev/cudatraining/triton_kernels/bench/bench_paged_attention_speed.py:1) — 10 shapes × 5 block sizes × {SDPA, paged} warmup=50 iter=200.
- 기준: `torch.nn.functional.scaled_dot_product_attention(..., enable_gqa=True)` — dispatcher 가 cuDNN / FA-2 / aten 중 고름.

결과 요약:

```
| shape               | B  | H  | H_kv | group | SDPA ms | best paged | gap    |
|---------------------|----|----|------|-------|---------|------------|--------|
| llama7b (MHA)       | 8  | 32 |   32 |   1   | 1.143   | 1.06x      | -7%    |  ✅ parity
| llama38b (GQA)      | 8  | 32 |    8 |   4   | 0.271   | 0.86x      | +16%   |  ⚠
| llama38b (GQA)      | 32 | 32 |    8 |   4   | 1.147   | 0.73x      | +37%   |  ⚠
| llama70b (GQA)      | 4  | 64 |    8 |   8   | 0.069   | 0.31x      | +217%  |  ❌
| llama70b (GQA)      | 8  | 64 |    8 |   8   | 0.533   | 0.46x      | +117%  |  ❌
| mqa (group=32)      | 16 | 32 |    1 |  32   | 0.062   | 0.07x      | +1316% |  💥
```

**관측**: gap 이 `GQA_GROUP_SIZE` 에 거의 선형 — group=4 → +16-37 %, group=8 → +117-227 %, group=32 → +1316 %.

**해석**: 그리드 `(B, H_q)` 는 GQA group 안의 모든 query head 가 독립적으로 full KV scan. group 내 **`GROUP_SIZE` 개의 query head 가 같은 KV block 을 DRAM 에서 중복 로드**. SDPA 는 contiguous 레이아웃이라 L2 prefetcher 가 이 중복을 흡수 (MQA 의 SDPA 는 **542 GB/s**, L4 DRAM peak 300 GB/s 의 **1.8×** — L2 reuse 증거), 우리는 block_table indirection 때문에 L2 prefetch 패턴이 깨져서 매번 DRAM.

결론: **grid 를 `(B, H_kv)` 로 바꾸고, program 안에서 Q 를 GQA 그룹 크기로 묶어서 처리해야 한다.** vLLM 논문은 이걸 "QUERIES_PER_KV" 로만 부르지 구조를 안 보여줌. Phase 4 에서 소스를 직접 읽어 확인.

세부 결과: [`docs/lesson_11_phase3_findings.md`](/Users/xavier/dev/cudatraining/docs/lesson_11_phase3_findings.md:1) Phase 3 섹션.

### Phase 3.5 — Grid refactor + silent TF32 bug (반나절)

**두 가지 변경**:

**(1) Grid `(B, H_q)` → `(B, H_kv)`** (메인 변경). 각 program 이 GQA group size 만큼의 query head 를 한 번에 처리:

```python
grid = (B, H_kv)                              # 프로그램 수 / GROUP 로 감소
q = tl.load(q_ptrs)                            # (GROUP, HEAD_DIM) — 2D tile
# 루프 안:
scores = tl.dot(q_scaled, tl.trans(k))         # (GROUP, BLOCK)  ← GQA 그룹 전체
acc += tl.dot(p.to(v.dtype), v)                # (GROUP, HEAD)   ← 그룹별 독립 softmax
```

`tl.dot` 의 `M≥16` 제약 때문에 GROUP≥4 일 때만 MMA path 로 가고 GROUP<4 (MHA, Mistral) 는 manual broadcast fallback. Threshold 결정 과정은 시행착오:
- 처음엔 `GROUP≥8` (fp16 MMA 의 안전권) → LLaMA-3-8B (group=4) 가 manual fallback 에 머물러 32KB SMEM 점유 → occupancy 손해.
- `GROUP≥4` 로 낮춤 → LLaMA-3-8B gap **+161 % → -14 %** (SDPA 를 이김).

**(2) `input_precision="ieee"` on fp32 `tl.dot`** (잘 안 보이는 변경).

`tl.dot(fp32, fp32)` 가 sm_80+ 에서 **기본 TF32** (10-bit mantissa). 처음엔 이게 문제가 아니었음 — Phase 3 의 manual broadcast fallback 이 순수 fp32 였으니까. 그런데 Phase 3.5 에서 grid 바꾸고 GROUP≥4 에 `tl.dot` 을 켜니까 **fp32 MQA** 케이스의 max diff 가 3.6e-07 → **4.1e-04** 로 뛰어오름.

이유: TF32 는 두 operand 을 10-bit mantissa 로 잘라서 곱함. MQA 의 `GROUP=16, BLOCK=16, HEAD=64` score 에 summation step 이 `64 + log2(BLOCK)` 번 누적되면 10-bit 절단 오차가 누적되어 softmax 의 max 후보 경계에서 4e-4 편향. `input_precision="ieee"` 로 강제하면 Triton 이 TF32 3-pass 스택 (+2 low-bit 보정) 을 써서 IEEE 수치를 재구성 → max diff 3.6e-07 로 복구.

이 버그는 **fp16 에선 안 보임** — fp16 MMA 의 native 가 IEEE fp16 이라 TF32 우회 없음. fp32 path 에만 함정.

**결과** ([`docs/lesson_11_phase3_findings.md`](/Users/xavier/dev/cudatraining/docs/lesson_11_phase3_findings.md:1) Phase 3.5 섹션):

| shape | Phase 3 gap | Phase 3.5 gap | Δ |
|---|---|---|---|
| llama7b MHA B=8 ctx=2k | -7 % | -7 % | 변화 없음 |
| llama38b GQA B=8 ctx=2k | +161 % | **-14 %** | SDPA 를 이김 |
| llama38b GQA B=32 ctx=2k | +86 % | +3 % | parity |
| llama70b GQA B=4 ctx=2k | -2 % | -1 % | 이미 이겼음 (L2 가 우연히 맞아떨어짐) |
| llama70b GQA B=8 ctx=4k | -1 % | -1 % | " |
| mqa group=32 B=16 ctx=4k | +1316 % | **+85 %** | 대폭 호전, 아직 SDPA 가 빠름 |

- Correctness: **32/32 PASS** 유지 (fp16 9.8e-4, fp32 3.6e-7).
- MQA 의 잔여 +85 % 는 **L2 reuse 구조적 한계** — SDPA 가 이 shape 에서 698 GB/s (DRAM 의 2.3×) 를 낼 수 있는 건 1 KV head × 4k tokens × 128 dim × 2 B fp16 = 1 MB 가 L2 48 MB 에 쉽게 들어가서 32 query heads 가 공유하기 때문. 우리 paged 는 block_table indirection 때문에 L2 prefetch 패턴이 안 서서 이 reuse 를 못 낸다. 닫으려면 **ctx 축 split-k** 필요 (vLLM v2 의 방식).

### Phase 4 — vLLM 실제 소스 읽고 diff (반나절)

- `git clone --depth=1 https://github.com/vllm-project/vllm /tmp/vllm` (HEAD commit `2463f00`).
- 읽은 파일 (`/tmp/vllm/` 내):
  - `csrc/attention/paged_attention_v1.cu` (186 lines) — 2023 오리지널 CUDA 런처
  - `csrc/attention/paged_attention_v2.cu` (196 lines) — split-k 버전
  - `csrc/attention/attention_kernels.cuh` (670 lines) — 실제 kernel body
  - `csrc/attention/attention_utils.cuh` (57 lines) — Qk_dot helper
  - `vllm/v1/attention/ops/triton_unified_attention.py` (1268 lines) — **현행 Triton replacement**

[`docs/lesson_11_phase4_vllm_diff.md`](/Users/xavier/dev/cudatraining/docs/lesson_11_phase4_vllm_diff.md:1) 에 axis-by-axis diff 테이블 + convergence/divergence 지점 정리.

**핵심 발견 세 가지**:

1. **내 Phase 3.5 디자인은 vLLM 의 Triton unified kernel 과 axis-for-axis 매치.**
   - 둘 다 `(..., H_kv)` 그리드
   - 둘 다 Q 를 `(GROUP, HEAD)` 2D tile 로 로드 후 `tl.dot`
   - 둘 다 fp32 accumulator 로 online softmax
   - **KV cache layout 도 동일**: `(num_blks, blk_size, H_kv, d)` — 나는 논문만 보고 골랐는데 vLLM Triton 포트와 정확히 일치. (CUDA v1 은 더 복잡한 `(NB, H_kv, d/x, BLK, x)` 로 vectorize.)

2. **vLLM 자체가 내가 한 것과 같은 refactor 를 거쳤다.**
   - 2023 년 CUDA v1: `dim3 grid(num_heads, num_seqs)` — **per-query-head**, 정확히 내 Phase 3 (느린 것).
   - 이 당시엔 대부분 MHA 라 group redundancy 가 없었음 + MQA 도 KV 가 작아서 L2 가 먹어줌.
   - LLaMA-2-7B-chat/LLaMA-3/Mistral 이 GQA 로 오면서 per-query-head grid 가 병목.
   - vLLM 이 Triton 으로 옮기면서 `(q_block, H_kv)` 로 restructure. **내가 Phase 3.5 에서 한 refactor 와 동일**, 같은 forcing function (GQA group ≥ 4) 때문에.
   - 교훈: 이건 내가 똑똑한 게 아니라, **같은 문제를 같은 툴로 풀면 같은 답에 수렴한다** 는 증거. "ecosystem 이 이미 한 refactor 를 miniature 로 재현" 이라는 스토리로는 충분.

3. **내가 한 것 중 vLLM 이 안 한 것 한 가지: `input_precision="ieee"`.**
   - vLLM 의 `triton_unified_attention.py` 의 `tl.dot` 은 precision 지정 없음. production 에서 fp16/bf16 만 돌리니까 문제 없음.
   - 누가 fp32 로 그 경로를 돌리면 내가 본 것과 같은 4e-4 오차가 날 것. 이건 엄밀히는 production 아닌 "lesson 이 fp32 correctness 도 테스트하는" 맥락에서만 중요.

4. **내가 못 한 것 한 가지: split-k over ctx.**
   - vLLM 의 v2 (`paged_attention_v2.cu`) 와 `kernel_unified_attention_3d` 가 ctx 축으로 partition (기본 512 토큰) 후 reduce kernel 로 softmax 재조합.
   - 내 MQA 잔여 gap (+85 %) 은 정확히 이걸 해결하는 형태. "Phase 4.5 blueprint" 로 Phase 4 문서에 설계만 남겨둠 (반나절 작업 예정, Lesson 12 입구로 이월).

### Phase 5 — 핸드오프 + 블로그 + README (이 Phase)

- 이 문서 — 핸드오프.
- [`docs/blog_draft_lesson_11_paged_attention.md`](/Users/xavier/dev/cudatraining/docs/blog_draft_lesson_11_paged_attention.md:1) — 블로그 초안.
- [`README.md`](/Users/xavier/dev/cudatraining/README.md:1) 에 lesson 11 섹션 추가.

---

## 2. 산출물

커널 + 드라이버:
- [triton_kernels/paged_attention.py](/Users/xavier/dev/cudatraining/triton_kernels/paged_attention.py:1) — 메인 Triton 커널 (275 lines). Phase 1+2+3.5 최종본.
- [triton_kernels/paged_attention_ref.py](/Users/xavier/dev/cudatraining/triton_kernels/paged_attention_ref.py:1) — PyTorch reference oracle (218 lines).

Bench:
- [triton_kernels/bench/bench_paged_attention.py](/Users/xavier/dev/cudatraining/triton_kernels/bench/bench_paged_attention.py:1) — correctness bench, 16 shape × 2 dtype.
- [triton_kernels/bench/bench_paged_attention_phase0.py](/Users/xavier/dev/cudatraining/triton_kernels/bench/bench_paged_attention_phase0.py:1) — Phase 0 ref smoke.
- [triton_kernels/bench/bench_paged_attention_speed.py](/Users/xavier/dev/cudatraining/triton_kernels/bench/bench_paged_attention_speed.py:1) — 10 shapes × 5 block_sizes, SDPA vs paged.

문서:
- 이 파일 — 핸드오프.
- [`docs/lesson_11_plan.md`](/Users/xavier/dev/cudatraining/docs/lesson_11_plan.md:1) — 원 계획.
- [`docs/lesson_11_phase3_findings.md`](/Users/xavier/dev/cudatraining/docs/lesson_11_phase3_findings.md:1) — Phase 3 + Phase 3.5 결과 (벤치 테이블, signatures 1-5, before/after).
- [`docs/lesson_11_phase4_vllm_diff.md`](/Users/xavier/dev/cudatraining/docs/lesson_11_phase4_vllm_diff.md:1) — vLLM 소스 diff + Phase 4.5 split-k blueprint.
- [`docs/blog_draft_lesson_11_paged_attention.md`](/Users/xavier/dev/cudatraining/docs/blog_draft_lesson_11_paged_attention.md:1) — 블로그 초안.

로그:
- `results/lesson11-phase{1,3,3.5}-*.log`

---

## 3. 핵심 숫자 (L4 sm_89, CUDA 12.9, torch 2.11.0+cu128, Triton 3.6.0, fp16, warmup=50, iters=200)

### 3.1 Correctness (Phase 1 + 2, 32/32 PASS)

| shape kind | B | H_q | H_kv | group | d | ctx | fp16 max diff | fp32 max diff |
|---|---|---|---|---|---|---|---|---|
| MHA | 1–4 | 32 | 32 | 1 | 128 | 16–2048 | 9.8e-04 | 3.6e-07 |
| LLaMA-3-8B GQA | 2 | 32 | 8 | 4 | 128 | 1024 | 2.4e-04 | 3.6e-07 |
| LLaMA-70B GQA | 4 | 64 | 8 | 8 | 128 | 가변 | 3.1e-05 | 3.6e-07 |
| MQA | 2 | 16 | 1 | 16 | 64 | 2048 | 1.9e-06 | 3.6e-07 |

### 3.2 Speed (Phase 3.5 최종)

| shape | B | H | H_kv | group | ctx | SDPA ms | paged ms (best bs) | gap |
|---|---|---|---|---|---|---|---|---|
| llama7b MHA | 8 | 32 | 32 | 1 | 2048 | 1.322 | 1.227 (bs=16) | **-7 %** |
| llama7b MHA | 8 | 32 | 32 | 1 | 8192 | 6.115 | 4.927 (bs=64) | **-19 %** |
| llama7b MHA | 32 | 32 | 32 | 1 | 2048 | 4.885 | 5.014 (bs=64) | +3 % |
| **llama38b GQA** | **8** | **32** | **8** | **4** | **2048** | **0.308** | **0.264 (bs=16)** | **-14 %** (beats SDPA) |
| llama38b GQA | 32 | 32 | 8 | 4 | 2048 | 1.163 | 1.197 (bs=128) | +3 % |
| llama70b GQA | 4 | 64 | 8 | 8 | 2048 | 0.049 | 0.048 (bs=128) | **-1 %** |
| llama70b GQA | 8 | 64 | 8 | 8 | 4096 | 0.532 | 0.526 (bs=16) | **-1 %** |
| mqa | 16 | 32 | 1 | 32 | 4096 | 0.048 | 0.089 (bs=128) | +85 % (L2 reuse 한계) |

LLaMA-3-8B 와 70B shape 에서 **cuDNN / FA-2 를 이기거나 parity**. MHA 는 realistic batch 에서 -7 % ~ -19 % (우리가 빠름). MQA 만 구조적 L2 reuse 때문에 +85 % 남음 — split-k 로만 닫힘.

### 3.3 GQA gap: Phase 3 vs Phase 3.5

| shape | Phase 3 | Phase 3.5 | 개선폭 |
|---|---|---|---|
| llama38b B=8 | +161 % | **-14 %** | 226 % 점 |
| llama38b B=32 | +86 % | +3 % | 83 % 점 |
| llama70b B=4 | -2 % | -1 % | 이미 이겼음 |
| llama70b B=8 | -1 % | -1 % | " |
| mqa group=32 | +1316 % | **+85 %** | 1231 % 점 |

---

## 4. 세 가지 교훈

### (a) **Correctness 가 통과해도 구조적 문제는 속도로만 보인다**

Phase 2 에서 GQA 32/32 PASS. fp16 max diff 3e-5 — 모델 공장 기준 완전히 안전. 만약 여기서 "GQA 구현 끝" 이라고 했으면 realistic shape 에서 SDPA 대비 **2-13 배 느린** 커널을 shipping 했을 것.

Phase 3 의 속도 벤치가 없었으면 이 실수를 몰랐다. "벤치 테이블 + gap 컬럼" 이 없으면 issue 가 숨는다. 단순히 allclose 가 통과하는 것 이상의 metric 이 필요.

**규칙**: 새 커널은 **reference 와 allclose + SDPA 와의 gap 을 같이** 리포트. allclose 만 보고 끝내지 말고.

### (b) **두 독립 버그가 연쇄로 숨을 수 있다**

이 세션에서 버그가 **두 번** 튀어나왔다:

1. **Grid design** (Phase 3 이 드러냄) — Phase 2 까진 invisible. Correctness 는 grid 설계에 무관하니까.
2. **Silent TF32** (Phase 3.5 에서 tile shape 바꾸자 드러남) — Phase 3 까진 invisible. Manual broadcast path 는 순수 fp32 였으니까.

버그 1 을 안 고쳤으면 2 가 안 나타남. 고치자마자 나타남. **버그가 다른 버그 뒤에 숨어 있을 수 있다** — "1 개 고쳤으니 끝" 이 아니다.

**규칙**: 큰 refactor 후에는 correctness bench 를 **반드시** 재실행. 새 path 에 숨은 새 버그를 잡기 위해.

### (c) **Paper 를 읽고 짜도 소스와 수렴한다 — 그게 신호**

이 세션에서 나는 **vLLM 소스를 Phase 3.5 완성 후에** 읽었다. Phase 3.5 의 grid 설계는 SOSP 논문 + Phase 3 의 속도 시그니처만 보고 결정. 그런데 vLLM 의 현행 Triton 포트와 axis-for-axis 매치.

교훈: **구조가 결정되는 지점은 HW + workload + tool 이고, 독립적으로 고민해도 같은 답**. 이건 "내가 똑똑한 것" 이 아니라 "맞는 답이 하나" 라는 것. 오히려 이 convergence 가 **"내 설계가 맞다"** 의 증거.

반대로 **내가 한 것 중 소스에 없는 것** (IEEE 강제) 은 production context 차이 — 그 차이를 명시적으로 기록하면 credible.

---

## 5. 함정 기록

### 함정 1: `tl.load(block_table)` 의 dtype

증상: paged 출력이 contiguous 와 맞지 않음. max diff 가 **0.5 이상** (어느 shape 에선 diff 가 1.0 근처) — allclose 절대 못 감.

원인: Triton 의 `tl.load(ptr)` 은 ptr 의 pointee dtype 으로 로드. `block_table` 를 torch 에서 int64 로 만들었는데 Triton 커널이 int32 로 읽으려 했거나, 반대 방향. 주소 스케일링 (`* sizeof(elem)`) 이 어긋나서 물리 block id 가 pointer 연산으로 쓰이는 순간 완전히 엉뚱한 위치.

해결: **torch 쪽에서 `block_table.to(torch.int32)` 로 정규화** + 커널에서도 load 후 `.to(tl.int64)` 로 승격 (stride 곱셈에서 overflow 방지). `paged_attention.py:119`.

교훈: vLLM 은 int32 가 공식 포맷. 다른 서빙 엔진과 호환하려면 int32 로 통일.

### 함정 2: `tl.dot` 의 최소 tile shape (M, N, K ≥ 16 for fp16)

증상: `grid=(B, H_kv)` 로 refactor 하자마자 MHA (GROUP=1) 케이스에서 **컴파일 에러**: "tl.dot requires M ≥ 16".

원인: `tl.dot(q, k.T)` 에서 `q: (GROUP, HEAD) = (1, 128)`. M=1 로는 fp16 MMA 가 돌지 않음.

해결: constexpr branching.
```python
if GQA_GROUP_SIZE >= 4 and BLOCK_SIZE >= 16:
    scores = tl.dot(q_scaled, tl.trans(k))
else:
    # manual broadcast for MHA (GROUP=1) and Mistral (GROUP=2)
    scores = tl.sum(q_f[:, None, :] * k_f[None, :, :], axis=2)
```

threshold 는 시행착오:
- `GROUP >= 8`: LLaMA-3-8B (group=4) 가 fallback → 32 KB SMEM 점유 → occupancy 손해.
- `GROUP >= 4`: LLaMA-3-8B 가 MMA path 에 진입 → gap +161 % → -14 %.
- `GROUP >= 2`: Mistral (group=2) 도 MMA. 아직 측정 안 했지만 predict 는 중간 호전.

교훈: `tl.dot` path 의 threshold 는 workload 따라 튜닝. 한 값으로 고정 안 됨.

### 함정 3: `tl.dot(fp32, fp32)` 의 **기본값은 TF32** (sm_80+)

증상: MQA fp32 케이스 max diff **4.1e-04**. fp16 은 통과하는데 fp32 만 깨짐. (정상적으로는 fp32 가 더 정확해야 한다.)

원인: Triton 3.x 의 `tl.dot` 은 sm_80+ 에서 **fp32 입력도 TF32 로 하향** (10-bit mantissa). Score tile 에 누적되면 softmax 의 max 결정이 4e-4 편향. `input_precision` 을 지정하지 않으면 기본이 이 TF32 경로.

해결: `IS_FP32` constexpr branching.
```python
if IS_FP32:
    scores = tl.dot(q_scaled, tl.trans(k), input_precision="ieee")
else:
    scores = tl.dot(q_scaled, tl.trans(k)).to(tl.float32)
```

IEEE 는 3-pass TF32 스택 (2 low-bit 보정) 이라 **3× 느림**. fp32 경로에만 쓰고 fp16 MMA 는 그대로. production 은 fp16/bf16 이라 이 함정이 잘 안 보임 — 내가 fp32 correctness 도 테스트해서 잡힘.

교훈: `tl.dot` 의 precision 은 **inputs 의 dtype 으로 안 정해짐**. `input_precision=` 을 명시. 특히 fp32 on sm_80+.

### 함정 4: `paged` 벤치 스크립트의 `set -e` 오탐

증상: 벤치 돌리면 모든 shape 결과가 다 출력되는데 마지막에 "exit 1" 로 실패 표시.

원인: 스크립트 말미의 `[[ "${DO_NCU}" == "ncu" ]] && echo ...` 가 `DO_NCU=speed` 일 때 short-circuit 으로 exit 1 을 return → `set -e` 가 propagate.

해결: 두 가지 중 하나 — (a) 맨 끝 구문을 `|| true` 로 감싸거나, (b) `if-else` 로 치환. 벤치 데이터 자체는 멀쩡.

교훈: `[[ ... ]] && <action>` 은 action 이 실행 안 되면 exit 1. bash `set -e` 와 함께 쓸 때 trap.

### 함정 5: 한 번 실패한 tarball 이 남아서 다음 세션 block

증상: `scripts/gcp_run_lessonXX.sh` 가 `mktemp -t cudatraining.XXX.tar.gz` 에서 "File exists" 로 실패.

원인: 이전 세션이 interrupt / OOM 으로 죽으면서 `/tmp/cudatraining.*.tar.gz` 가 남음. 새 mktemp 이 충돌.

해결: 세션 시작 전에 `rm -f /tmp/cudatraining.*.tar.gz` 한 번 (또는 스크립트 안에서 trap 으로 cleanup).

교훈: GCP runner 에 `trap "rm -f /tmp/cudatraining.*" EXIT` 를 기본으로 두자.

### 함정 6: `tl.load` 의 mask 없이 마지막 partial block 로드

증상: ctx=777 같은 block_size 배수 아닌 경우 마지막 block 의 쓰레기 토큰이 softmax 에 기여 → NaN 이거나 편향.

원인: block_size=16, ctx=777 이면 마지막 block 이 9 토큰만 유효. 나머지 7 토큰은 쓰레기 (할당만 된 blob). 로드할 때 mask 안 걸면 softmax 에 들어감.

해결:
```python
token_idx = logical_blk * BLOCK_SIZE + offs_n
mask_n = token_idx < ctx_len
k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
...
scores = tl.where(mask_n[None, :], scores, -float("inf"))
```

두 번 걸어야 함 — load 시 K/V 에 mask, score 에 `-inf` mask. `other=0.0` 만으로는 부족 (Q·0=0 이지만 softmax exp(0)=1 이 pollute).

Phase 2 의 case 13 (B=4, ctx=[256,1024,2048,777]) 이 이 함정을 잡아줬다.

---

## 6. 다음 세션에 남기는 것

1. **Phase 4.5-a (ncu drill)**: MQA 의 L2 reuse 가설을 숫자로 고정. `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` + `lts__t_sectors_*` 로 DRAM vs L2 read volume 비교. 가설: SDPA 는 L2 hit rate 90+ %, 우리는 20-30 %. 반나절 작업, 필요하면 lesson 10 의 sudo -E ncu 스크립트 재사용.

2. **Phase 4.5-b (split-k 커널)**: ctx 축 partition + reduce kernel. [`lesson_11_phase4_vllm_diff.md`](/Users/xavier/dev/cudatraining/docs/lesson_11_phase4_vllm_diff.md:252) 의 blueprint 그대로. 기대: MQA gap +85 % → +20 % 내. 하루 작업.

3. **Lesson 12 후보**: prefill (long q_len) 지원해서 vLLM `kernel_unified_attention_3d` 의 `BLOCK_Q` 패킹 재현. 여기에 split-k 까지 있으면 "decode + prefill + split-k 를 한 커널로" 의 v1 수준 완성.

4. **FP8 KV cache**: vLLM production 에서 실제 deployment 의 30 % 가 fp8. per-head scale 지원 + tl.dot 의 fp8 path. Triton 3.x 에서 실험적 지원 있음.

---

## 7. 요약 한 문단

> Lesson 09 의 contiguous MHA 를 vLLM-style paged KV cache 로 재작성했다. correctness 32/32 통과 후
> **속도 벤치에서 GQA group 크기에 비례하는 구조적 gap 을 발견** — grid `(B, H_q)` 가 각 query
> head 마다 KV 를 중복 로드한다는 문제. Phase 3.5 에서 grid 를 `(B, H_kv)` 로 refactor 해서 LLaMA-3-8B
> gap 을 **+161 % → -14 % (SDPA 를 이김)** 로 닫았고, 그 과정에서 `tl.dot(fp32, fp32)` 이 기본으로
> TF32 로 떨어지는 함정 (4e-4 오차) 을 잡아서 `input_precision="ieee"` 로 고정. vLLM 의 현행 Triton
> unified kernel 을 Phase 3.5 완성 후 읽어보니 axis-for-axis 매치 — **vLLM 자신이 2023 년 CUDA v1
> 에서 per-query-head 로 시작했다가 GQA 모델이 shipping 되면서 같은 refactor 를 했음**. 남은 MQA +85 %
> 잔여 gap 은 L2 reuse 구조 한계이고, vLLM 의 v2 에 해당하는 ctx 축 split-k 로만 닫힘 — Lesson 12 로
> 이월. 이번 세션의 알맹이: (1) correctness 가 통과해도 구조 버그는 속도로만 보인다, (2) 한 번에
> 두 독립 버그가 연쇄로 숨을 수 있다, (3) paper + HW + workload 만으로 짜도 소스와 수렴하면 그게
> 설계의 증거.
