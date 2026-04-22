# Lesson 12 Handoff — Paged Attention 의 MQA Gap 을 vLLM v2-style Split-K 로 닫기

기준 날짜: `2026-04-22`

주제: **Lesson 11 Phase 4 에서 남긴 "MQA 에서 SM 이 놀고 있다" 라는 구조적 결함을 vLLM v2 의 split-k + reduce 2-kernel 아키텍처로 정리**. 새 블로그용 커널이 아니라, **이미 짠 커널의 아키텍처 구멍을 닫는 세션**.

세션 규모: 4 Phase (1 = split + reduce 커널 + auto-dispatch 래퍼, 2 = correctness, 3 = speed bench + heuristic 튜닝, 4 = 이 문서 + 블로그 + README).

---

## 1. 이번 세션에서 한 일

### 왜 이 세션이 필요한가

Lesson 11 Phase 4 의 한 줄 결론:

> GQA (group≥4) 에서는 vLLM 보다 빠름. **MQA 에서는 B\*H_kv 가 너무 작아서 SM 이 놀고 있음.** 이건 우리 커널의 구조적 결함이고, vLLM 은 v2 에서 split-k 로 해결함.

벤치 숫자로:
```
| 모양                         | SP gap vs SDPA |
|------------------------------|---------------|
| LLaMA-3-8B B=8 ctx=2k GQA   | +21.5%        |  (괜찮음)
| LLaMA-70B B=8 ctx=4k GQA=8  | +11.0%        |  (괜찮음)
| MQA  B=16 H_kv=1 ctx=4k     | +645.8%       |  ← 이 세션의 타깃
```

MQA 에서 B\*H_kv = 16. L4 는 58 SM. 16 개 프로그램으로는 SM 점유율 28 %. 남은 42 SM 은 놀고 있음. Single-pass 그리드 `(B, H_kv)` 만으로는 이 구멍을 못 닫는다.

vLLM 의 `paged_attention_v2.cu` 가 정확히 이 문제를 풀기 위해 존재. 우리가 Lesson 11 에서 읽은 대로:
- Grid: `(num_heads, num_seqs, max_num_partitions)`, PARTITION_SIZE = 512 tokens
- Each block handles PARTITION_SIZE tokens of one sequence
- Separate reduce kernel recombines partial (m, l, acc) via 표준 online-softmax 공식

이 세션의 목적은 한 줄:

> vLLM v2 의 2-kernel (split-k + reduce) 아키텍처를 Triton 으로 재현. MQA gap 이 줄어드는지 확인. 자동 dispatch 휴리스틱 튜닝.

### Phase 1 — split kernel + reduce kernel + auto-dispatch wrapper (반나절)

- [triton_kernels/paged_attention.py](/Users/xavier/dev/cudatraining/triton_kernels/paged_attention.py:1) — 기존 `paged_attention_decode_kernel` (single-pass) 유지. **새 함수 2 개 추가**:
  - `paged_attention_split_kernel` — grid `(B, H_kv, SEGMENTS)`. 각 program 이 자기 segment 의 `PARTITION_SIZE = 512` 토큰만 처리. **UNNORMALIZED** online softmax 상태 `(m_i, l_i, acc)` 를 scratch 에 기록.
  - `paged_attention_reduce_kernel` — grid `(B, H_q)`. SEGMENTS 축을 다 로드해서 `alpha = exp(m_s - m_global)` 로 recombine. 최종 normalized output 을 native dtype 으로 저장.

- Wrapper 에 `use_split_k: bool | None = None`, `partition_size: int = 512` 추가.

- 분기 로직: `use_split_k=None` 이면 auto-dispatch. `True`/`False` 는 진단용 강제 스위치.

커널 공유 구조:
- split_kernel 의 **inner loop** 는 single-pass 와 동일 — K/V 로드 + 스코어 + online softmax rescale. 차이는 **loop range** 뿐: single-pass 는 `range(0, num_blocks_total)`, split 은 `range(seg_block_start, seg_block_end)`.
- 덕분에 Phase 3.5 에서 쌓아놓은 `tl.dot` + `input_precision="ieee"` 최적화가 그대로 녹아들어옴. 복붙은 좀 찝찝했지만 함수 1 개당 constexpr 4 개 (BLOCK_SIZE, HEAD_DIM, GQA_GROUP_SIZE, IS_FP32) 의 specialization 이 걸려서 완전 공통화는 현실적으로 어려움.

### Phase 2 — Correctness (30 분)

Bench 에 split-k 강제 호출 추가: `bench_paged_attention.py` 의 16 shape × 2 dtype = 32 case 각각 **single-pass 와 split-k 두 번 호출**, 둘 다 `naive_decode_attention` 와 매칭되는지 + 서로 매칭되는지 확인.

**첫 실행: 28/32 PASS**. fp16/fp32 각각 cases 7, 12 에서 실패:
```
CompilationError: arange's range must be a power of 2
```

Case 7: `B=2, H=32, d=128, context_lens=[513, 129], block_size=16`. `partition_size=32` (블록 2 배), ctx_max=513 → segments = ⌈513/32⌉ = **17**. Triton `tl.arange(0, SEGMENTS)` 는 SEGMENTS 가 power of 2 여야 함.

**수정** (paged_attention.py):
- `_next_pow2(n)` 헬퍼. `reduce_kernel` 시그니처에 `SEGMENTS_P2: tl.constexpr` 추가.
- 커널 본문: `offs_s = tl.arange(0, SEGMENTS_P2)`, `mask_s = offs_s < SEGMENTS`. 패딩 lane 은 `other=-float("inf") / 0.0` 로 로드 → `exp(-inf - m_global) = 0` 로 자연스럽게 캔슬.
- Scratch 할당은 `segments` 크기 그대로 (padding 은 kernel 쪽에서만 처리).

**재실행: 32/32 PASS** (single-pass: 32/32, split-k: 32/32, split-k 이 실제로 2 segment 이상으로 분할된 경우 22/32).

diff 최대값 (fp16):
- single-pass vs naive: 9.77e-04
- split-k vs naive:    9.77e-04
- **split-k vs single-pass: 9.54e-07 ~ 4.88e-04** (경우마다 다름, 둘 다 독립적으로 online softmax 함, 수치 오차 범위 일치)

### Phase 3 — Speed bench + heuristic 튜닝 (반나절)

`bench_paged_attention_speed.py --compare-paths`: 10 shape × {SP, SK, auto} at block_size=16, partition_size=512.

**첫 실행** (heuristic 임시: `B*H_kv < 0.75*SM AND segments >= 2`):

```
| shape              | SP gap | SK gap | auto gap | auto 선택 |
|--------------------|--------|--------|----------|----------|
| llama7b-B1-ctx1k   | +316%  | +480%  | +483%    | SK (나쁨) |
| llama7b-B1-ctx4k   |  +82%  |  +49%  |  +49%    | SK (좋음) |
| llama70b-B4-ctx2k  | +283%  | +368%  | +369%    | SK (나쁨) |
| mqa-B16-ctx4k      | +646%  | +350%  | +353%    | SK (좋음) |
```

**관찰**:
1. MQA 에서 SK 가 SP 대비 **1.68× 빠름** (0.331 → 0.197 ms) — 구조적 수정이 먹힘.
2. 하지만 휴리스틱이 llama70b-B4-ctx2k (B\*H_kv=32, group=8) 에서도 SK 를 골랐는데, 여기서는 **SP 가 더 빠름**. GROUP 이 클 때는 single-pass 가 이미 KV 로드를 amortize 하고 있어서 SK 의 추가 parallelism 은 reduce 커널 launch overhead 만 더함.
3. llama7b-B1-ctx1k (B\*H_kv=32, segments=2) 에서도 SK 가 SP 보다 느림. Segments=2 는 amortization 이 안 됨.

**휴리스틱 수정**:
```python
# Before
use_split_k = (B*H_kv < SM*0.75) AND (segments >= 2)
# After
use_split_k = (B*H_kv < SM*0.5) AND (segments >= 4)
```

**재실행** (튜닝 후):

```
| shape              | SP ms  | SK ms  | auto ms | auto 선택 | 
|--------------------|--------|--------|---------|----------|
| llama7b-B1-ctx1k   | 0.142  | 0.196  | 0.143   | SP ✓    |  (fix)
| llama7b-B1-ctx4k   | 0.475  | 0.389  | 0.466   | SP       |  (miss, but small)
| llama7b-B8-ctx2k   | 1.140  | 1.206  | 1.144   | SP ✓    |
| llama7b-B32-ctx2k  | 4.359  | 4.448  | 4.361   | SP ✓    |
| llama7b-B8-ctx8k   | 4.370  | 4.411  | 4.367   | SP ✓    |
| llama38b-B8-ctx2k  | 0.326  | 0.393  | 0.327   | SP ✓    |
| llama38b-B32-ctx2k | 1.130  | 1.227  | 1.129   | SP ✓    |
| llama70b-B4-ctx2k  | 0.165  | 0.196  | 0.166   | SP ✓    |  (fix)
| llama70b-B8-ctx4k  | 0.591  | 0.693  | 0.591   | SP ✓    |
| mqa-B16-ctx4k      | 0.331  | 0.196  | 0.197   | SK ✓    |  (-41% vs SP)
```

10 shape 중 9 shape 에서 auto 가 정답 선택 (llama7b-B1-ctx4k 에서 SP 를 고르지만 SK 가 17 % 빠름 — 허용). **가장 큰 이득**: MQA 에서 SP 대비 **1.68× speedup** (0.331 → 0.197 ms).

### Phase 4 — docs + commit

이 문서 + `blog_draft_lesson_12_split_k.md` + README Lesson 12 라인 + commit.

---

## 2. 숫자 요약 — Before / After

**target shape**: `mqa-B16-ctx4k` (B=16, H=32, H_kv=1, d=128, ctx=4096, block_size=16, fp16).

| 측정 | Lesson 11 Phase 4 종료 시점 | Lesson 12 종료 시점 |
|---|---|---|
| SDPA | 0.044 ms | 0.044 ms |
| paged single-pass | 0.331 ms  | 0.331 ms |
| paged **split-k** | (구현 없음) | **0.196 ms** |
| paged auto | = single-pass | **0.197 ms** (=split-k) |
| MQA gap vs SDPA | **+645.8 %** | **+344.1 %** |
| MQA paged speedup | 1.00× | **1.68×** |

다른 9 shape 는 auto-dispatch 가 single-pass 를 계속 고르므로 변화 없음. Regression 없음.

---

## 3. 이번 세션에서 배운 것

### 배움 1 — "이미 느리다" 는 아키텍처 문제의 증거

Lesson 11 Phase 4 에서 관찰된 MQA +645 % gap 은 "아직 덜 튜닝했다" 가 아니라 **구조적 결함** 이었음. Block size 를 8 ~ 128 까지 쓸어도 최저 0.203 ms (bs=32) 로 SDPA 0.044 ms 에 근처도 못 감. 파라미터 튜닝으로 닫을 구멍이 아니었음. **vLLM 이 v1 → v2 에서 같은 구멍을 본 증거** (code comment 에 "partition_size = 512 means we split...").

튜닝을 먼저 해보고 안 되면 아키텍처를 의심, 이 순서가 틀린 건 아니지만, **metric 이 parameter 스윕에 반응하지 않는 시점이 결정적 신호** 라는 걸 다시 확인.

### 배움 2 — Triton `tl.arange` pow-of-2 제약은 mask 로 깔끔하게

초기에는 "scratch 를 pow2 로 padding 해야 하나" 고민했음 (segments=17 → 32 할당). 근데 그러면 scratch 메모리 낭비 + grid_fwd 도 같이 커져야 함.

더 나은 방법: **kernel constexpr 만 SEGMENTS_P2 로, scratch 는 actual SEGMENTS 로**. kernel 안에서 `mask_s = offs_s < SEGMENTS` 로 패딩 lane 무시. Masked `tl.load` 의 `other=-inf` 를 쓰면 **재결합 수식이 자동으로 0 을 기여** (`exp(-inf - m_global) = 0`).

이 아이디어는 online softmax 의 구조 자체에 녹아있음 — "invalid segment" 와 "empty segment" 가 수학적으로 같은 처리를 받음. **sentinel 값이 무시 동작과 수학적으로 동등** 한 자리는 코드를 단순하게 만든다.

### 배움 3 — 휴리스틱은 regression 안 나게 튜닝해야 함

첫 heuristic 은 "SK 가 도와주는 곳에 dispatch" 를 과도하게 공격적으로 잡음 (0.75 · SM). 결과: 몇몇 shape 에서 SK 가 regression 을 만듦.

두 번째 heuristic (0.5 · SM, segments >= 4): **SK 가 확실히 좋은 곳에만 dispatch**. 대신 llama7b-B1-ctx4k 같은 "조금 좋아지는" 케이스를 놓침. 이건 tradeoff 상 맞음 — 자동 dispatch 는 **regression 안 내는 게 첫 번째 원칙**, 이득은 두 번째.

실전 커널의 auto-tuning 도 같은 원칙: "의심스러우면 기본(단일-패스)으로" 가 보수적이지만 옳다.

### 배움 4 — SDPA 가 "이길 수 없이 빠를 때" 의 정체: L2 locality

MQA 기준 SDPA 0.044 ms 는 L4 의 DRAM 피크 300 GB/s 기준으로 보면 bytes_moved / 0.044 ms = **761 GB/s 등가**. 물리적으로 불가능.

답: **L2 hit**. B=16, H_kv=1, ctx=4096, d=128, fp16 → K/V 총량 = 16·1·4096·128·2 = **16 MB**. L4 의 L2 = 48 MB → **완전 L2 resident**. SDPA 의 contiguous layout 이 L2 에서 다시 읽으니까 300 GB/s 의 2.5× 처럼 보이는 것.

우리 paged 는:
1. `block_table` lookup (indirection) 가 L2 cache line 을 추가로 씀.
2. block 단위 gather 가 L2 의 spatial locality 를 깨뜨림.
3. 결과: paged 는 DRAM bound (165 GB/s 실제), SDPA 는 L2 bound (761 GB/s 등가).

이 gap 을 닫으려면 L2 prefetch / pinned block 배치 같은 더 깊은 최적화 필요. Split-k 는 **SM 점유율** 을 고치지, **L2 residency** 를 고치지 않음. MQA 는 근본적으로 L2 문제라는 결론이 이 세션의 한계 (= 다음 세션 거리).

---

## 4. 함정 / 디버그 기록

| # | 함정 | 증상 | 원인 | 해결 |
|---|---|---|---|---|
| 1 | `tl.arange` pow-of-2 | `CompilationError: arange's range must be a power of 2` | `segments = ceil(513/32) = 17` 이 pow2 아님 | `SEGMENTS_P2 = next_pow2(SEGMENTS)`, mask load |
| 2 | auto-dispatch over-trigger | `llama70b-B4` 에서 auto=SK 가 SP 보다 20 % 느림 | heuristic `B*H_kv < 0.75*SM AND segments >= 2` 너무 관대 | `0.5*SM` + `segments >= 4` |
| 3 | scratch 낭비 걱정 | 초기에 "pow2 만큼 할당해야 하나" 걱정 | `tl.arange` 가 pow2 필요해서 | scratch 는 actual SEGMENTS, kernel 은 SEGMENTS_P2 + mask |
| 4 | invalid segment 처리 | "segments 끝까지 도달 안 한 프로그램은 뭐 쓰지?" | 그냥 초기 `m=-inf, l=0, acc=0` 을 기록하면 됨 | reduce 의 `exp(-inf - m_global) = 0` 로 자동 캔슬 |

---

## 5. 남긴 것 (= 다음 세션 거리)

1. **L2-locality aware paged**: MQA 에서 SDPA 의 L2 residency 를 어떻게 흉내 낼지. Pinned block layout? Per-SM L1 staging? 이 깊어지면 lesson 13 수준.
2. **Prefill kernel**: decode 만 했음. Prefill (Q 가 긴 시퀀스) 용 paged kernel 은 또 다른 story (block-sparse, causal mask). vLLM 도 prefill 과 decode 를 분리함.
3. **bf16 경로**: 현재 fp16 / fp32. bf16 은 mantissa 7-bit 라서 online softmax 의 scale factor rounding 주의.
4. **Autotuning**: partition_size=512 는 고정. 긴 ctx (16k+) 에서는 1024 가 더 나을 수도. `@triton.autotune` 으로 {256, 512, 1024} 스윕.

다음 세션 후보:
- (a) vLLM 의 unified attention kernel (decode + prefill 같은 커널) 를 Triton 으로 재현 — 레슨 11+12 를 녹여서 vLLM PR 준비.
- (b) 뭔가 다른 커널 카테고리 — MoE gate + gather + matmul 같은 LLM serving 쪽 실전 커널.

---

## 6. 파일 변경 요약

### 수정
- `triton_kernels/paged_attention.py` — 기존 single-pass 커널 유지, **split_kernel + reduce_kernel 2 개 추가**, wrapper 에 auto-dispatch + `use_split_k` + `partition_size` 추가. ~130 줄 추가, 총 ~550 줄.
- `triton_kernels/bench/bench_paged_attention.py` — 각 case 에서 SP + SK 두 번 호출, 패스/페일 둘 다 리포트.
- `triton_kernels/bench/bench_paged_attention_speed.py` — `--compare-paths` flag, `run_path_compare()` + `print_path_compare()` 추가.
- `README.md` — Lesson 12 행 + headline result 업데이트.

### 신규
- `docs/lesson_12_handoff_2026-04-22.md` — 이 문서.
- `docs/blog_draft_lesson_12_split_k.md` — 블로그 초안.
- `scripts/gcp_run_lesson12.sh` — correctness + speed bench runner.

---

## 7. 한 문단 요약

Lesson 11 Phase 4 에서 "MQA 에서 B\*H_kv=16 이 L4 의 58 SM 을 다 채우지 못한다" 는 구조적 결함을 관찰했다. 이 세션에서는 vLLM 의 paged_attention_v2 가 정확히 이 문제를 풀기 위해 도입한 **split-k + reduce 2-kernel 아키텍처** 를 Triton 으로 재현했다. split 커널은 ctx 축을 `PARTITION_SIZE=512` 로 쪼개 `(B, H_kv, SEGMENTS)` 그리드로 올리고, 각 프로그램이 UNNORMALIZED `(m, l, acc)` 을 scratch 에 기록한다. reduce 커널이 SEGMENTS 축을 `alpha = exp(m_s - m_global)` 로 재결합한다. `tl.arange` 의 power-of-2 제약은 scratch 를 padding 하지 않고 kernel constexpr 로 `SEGMENTS_P2` 를 분리 + mask-based load 로 해결했다. Auto-dispatch 휴리스틱은 `B*H_kv < 0.5·SM AND segments >= 4` — 첫 시도 (`0.75·SM AND segments >= 2`) 가 LLaMA-70B B=4 shape 에서 SK 를 고르며 오히려 20 % 느려지는 regression 을 내서 조였다. 결과: **MQA paged 가 1.68× 빨라짐** (0.331 → 0.197 ms). SDPA 대비 gap 은 여전히 +344 % 로 큰데, 그 원인은 SM 점유율이 아니라 **L2 residency** (K/V 16 MB 가 L4 의 48 MB L2 에 통째로 들어가서 SDPA 는 DRAM 이 아니라 L2 에서 읽음). 이 gap 은 split-k 로 닫을 수 없는 영역이고, 다음 세션 거리로 남겼다.
