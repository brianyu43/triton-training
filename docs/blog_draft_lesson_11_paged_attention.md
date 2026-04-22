# vLLM 의 Paged Attention 을 Triton 으로 다시 짜보고, vLLM 의 실수를 반복했다

*— 독립적으로 같은 refactor 에 수렴하는 게 설계가 맞다는 증거라는 이야기.*

기준 날짜: 2026-04-22  ·  GPU: NVIDIA L4 (Ada Lovelace, sm_89, 24GB)  ·  torch 2.11.0+cu128 · Triton 3.6.0 · fp16/fp32

---

## 들어가며 — contiguous attention 에 끼워넣는 한 줄

Lesson 09 에서 `(B, H, N, d)` 4-D MHA + causal flash attention 을 Triton 으로 짰다. 그 커널은 `torch.nn.functional.scaled_dot_product_attention` (속을 까보면 Tri Dao 의 FA-2 CUDA) 의 **78-90 %** 속도로 돌아간다. 벤치 수치만 보면 "잘 짠 커널" 이다.

문제는 **LLM 서빙 (vLLM / SGLang / TensorRT-LLM) 은 contiguous KV cache 를 쓰지 않는다.** sequence 별로 길이가 천차만별이고, 들어왔다 나갔다 하니까 pre-allocate 하면 GPU 메모리 70 % 이상이 fragmentation 으로 날아간다. vLLM 이 SOSP '23 에서 밀어붙인 해법은 단순:

**KV cache 를 고정 크기 block 의 pool 로 쪼개고, sequence 별 `block_table` 로 간접 참조한다.**

```
(기존)
K: (B, H, N, d)                             ← seq 별 연속

(paged)
K_cache:     (num_blocks, block_size, H_kv, d)     ← 블록 풀
block_table: (B, max_blocks_per_seq)               ← seq → physical block id
context_lens:(B,)                                   ← 각 seq 유효 길이
```

attention 커널이 해야 할 일이 바뀌는 건 단 한 줄:

```python
# 기존: K[b, h, start_n:end_n, :] 를 연속으로 로드
# paged: block_table[b, logical_blk] 조회 → phys_blk →
#        K_cache[phys_blk, :, kv_head, :] 로드
```

Lesson 11 은 **"이 한 줄만 끼워넣으면 vLLM paged attention 이 되는지"** 를 측정하는 세션. 되긴 되는데, 가는 길에 **두 번** 틀렸다. 그게 이 글의 알맹이다.

---

## Phase 1-2 · correctness 는 너무 쉽게 통과했다 (이게 함정)

작업 순서:
1. PyTorch reference (Python loop 로 block_table 돌면서 gather 후 표준 attention) 로 oracle 만들기.
2. Triton 커널: 처음엔 `grid = (B, H_q)` — lesson 09 의 `(cdiv(N, BM), H, B)` 에서 "decode 니까 N=1" 로 자연스럽게 축소.
3. GQA 지원: `GQA_GROUP_SIZE = H_q // H_kv` 를 constexpr 로 추가, `kv_head = pid_h // GQA_GROUP_SIZE` 로 KV 참조. **총 4 줄 변경**.

Correctness 벤치 (16 shape × 2 dtype = **32/32 PASS**):

| shape | B | H_q | H_kv | group | fp16 max diff | fp32 max diff |
|---|---|---|---|---|---|---|
| MHA | 1-4 | 32 | 32 | 1 | 9.8e-04 | 3.6e-07 |
| LLaMA-3-8B GQA | 2 | 32 | 8 | 4 | 2.4e-04 | 3.6e-07 |
| LLaMA-70B GQA | 4 | 64 | 8 | 8 | 3.1e-05 | 3.6e-07 |
| MQA | 2 | 16 | 1 | 16 | 1.9e-06 | 3.6e-07 |

fp16 에서 1e-3 이하, fp32 에서 1e-7 수준. 모델 공장 기준 모든 게 통과.

이 시점의 판단: **"되는구나. 끝나겠다."**

이게 함정이었다.

---

## Phase 3 · 속도 벤치가 구조 버그를 드러냈다

Correctness 만 보고 끝냈으면 이 커널이 LLaMA-3-8B 에서 SDPA 대비 **2-3 배 느린 상태로** shipping 됐을 것. 속도 벤치를 돌려본 결과:

```
| shape              | B  | group | SDPA ms | paged best | gap     |
|--------------------|----|-------|---------|------------|---------|
| llama7b (MHA)      | 8  |   1   | 1.143   | 1.06x      |  -7%    |  ✅
| llama38b (GQA)     | 8  |   4   | 0.271   | 0.86x      | +16%    |  ⚠
| llama38b (GQA)     | 32 |   4   | 1.147   | 0.73x      | +37%    |  ⚠
| llama70b (GQA)     | 4  |   8   | 0.069   | 0.31x      | +217%   |  ❌
| llama70b (GQA)     | 8  |   8   | 0.533   | 0.46x      | +117%   |  ❌
| mqa                | 16 |  32   | 0.062   | 0.07x      | +1316%  |  💥
```

**MHA 는 parity (SDPA 와 동률 또는 빠름). GQA 부터 gap 이 `GROUP_SIZE` 에 거의 선형.**
- group=4 → +16-37 %
- group=8 → +117-217 %
- group=32 → **+1316 %** (13× 느림)

이 선형성이 진단의 핵심. 임의 regression 이 아니라 **구조적**.

### 원인 — `(B, H_q)` 그리드가 KV 를 GROUP 번 재로드한다

Grid `(B, H_q)` 는 한 program 이 한 `(batch, query head)`. GQA 의 `GROUP_SIZE` 개의 query head 는 **같은 KV head** 의 캐시를 공유하지만, 각각의 program 이 독립적으로 block_table 를 돌며 **같은 K/V block 을 DRAM 에서 다시** 로드한다. `GROUP_SIZE` 배의 redundant DRAM 트래픽.

SDPA 는 왜 안 느리나? Contiguous KV 니까 **L2 prefetcher 가 중복 로드를 흡수**한다. 실제로 MQA 의 SDPA 는 L4 에서 **542 GB/s** — L4 DRAM peak (300 GB/s) 의 1.8 배. 이 숫자는 DRAM 단독으로 불가능. L2 가 반 이상을 먹고 있다는 증거.

우리 paged 는 block_table indirection 때문에 L2 prefetcher 가 패턴 인식을 못한다 (한 sequence 의 logical block 들이 memory 상에선 무작위 물리 offset). 그래서 redundant load 가 전부 DRAM 으로.

**교훈 #1**: **Correctness 가 통과해도 구조 문제는 속도로만 보인다.** `allclose` 는 grid 설계에 무관하다. Reference 와의 비교만으론 이 버그는 절대 안 잡혔다. 벤치 테이블 + SDPA gap 컬럼을 리포트로 남겨야 문제가 보인다.

---

## Phase 3.5 · Grid 하나 바꿨을 뿐인데 (또 다른 버그가 튀어나왔다)

고칠 게 명확하다. **Grid 를 `(B, H_kv)` 로 바꾸고, program 안에서 GQA group 의 `GROUP_SIZE` query head 를 한 번에 처리한다**. K/V block 은 program 당 한 번만 로드.

```python
grid = (B, H_kv)                              # 프로그램 수 / GROUP
q = tl.load(q_ptrs)                            # (GROUP, HEAD_DIM) — 2D tile
# 블록 루프 안:
scores = tl.dot(q_scaled, tl.trans(k))         # (GROUP, BLOCK)
acc += tl.dot(p.to(v.dtype), v)                # (GROUP, HEAD)
```

이 변경 자체는 ~20 줄. 벤치 돌려보니:

| shape | Phase 3 gap | Phase 3.5 gap |
|---|---|---|
| llama38b (group=4) B=8 | +161 % | **-14 %** ← SDPA 를 이김 |
| llama38b (group=4) B=32 | +86 % | +3 % (parity) |
| llama70b (group=8) B=4 | -2 % | -1 % |
| llama70b (group=8) B=8 | -1 % | -1 % |
| mqa (group=32) | +1316 % | **+85 %** |

LLaMA-3-8B 의 `-14 %` — cuDNN / FA-2 를 우리 Triton 커널이 **이기는** shape 가 생겼다. 이 shape 는 production 에서 흔한 mid-range batch.

### 그런데 fp32 correctness 가 깨졌다

Correctness 재실행:
- fp16: 모든 shape 여전히 통과 ✅
- fp32 MQA: max diff **4.1e-04** ← **원래 3.6e-07 이었다**. 세 자릿수 나쁨.

당황. grid 만 바꿨는데 fp32 가 왜 깨지나?

### 진짜 원인 — `tl.dot(fp32, fp32)` 의 **기본값은 TF32** (sm_80+)

Ampere 이상에서 Triton 은 `tl.dot` 의 fp32 × fp32 를 **자동으로 TF32 로 하향** (10-bit mantissa). `input_precision` 을 명시 안 하면 기본이 TF32. MQA 의 `(GROUP=16, BLOCK=16, HEAD=64)` score tile 에서 summation 이 80-100 번 누적되면 10-bit 절단 오차가 쌓여서 softmax max 후보 경계가 4e-4 편향.

해결:
```python
if IS_FP32:
    # 3-pass TF32 스택 (2 low-bit 보정) 으로 IEEE 재구성 — 3× 느림
    scores = tl.dot(q_scaled, tl.trans(k), input_precision="ieee")
else:
    # fp16/bf16 MMA — default 는 이미 IEEE fp16
    scores = tl.dot(q_scaled, tl.trans(k)).to(tl.float32)
```

fp32 max diff: **4.1e-04 → 3.6e-07** 복구. fp16 speed 손해 없음 (fp16 path 는 그대로).

### 왜 Phase 3 에선 안 보였나

Phase 3 은 manual broadcast (`tl.sum(q * k)`) 로 score 를 계산했고, 이건 **순수 fp32** path. TF32 거치지 않음. Phase 3.5 에서 `tl.dot` 을 도입한 순간 처음 노출된 버그.

**교훈 #2**: **두 독립 버그가 연쇄로 숨을 수 있다.** Grid bug 를 안 고쳤으면 TF32 bug 가 안 나타남. 고치자마자 나타남. 큰 refactor 뒤엔 correctness 를 **반드시** 재실행. 한 버그 고쳤다고 끝이 아니다.

---

## Phase 4 · vLLM 소스 읽고 보니 나는 vLLM 역사를 miniature 로 재현했다

Phase 3.5 가 끝난 뒤에야 vLLM 소스를 읽었다. 일부러 — **독립적으로 설계한 뒤 vLLM 과 비교해서 수렴하는지** 보고 싶었다.

`git clone --depth=1 https://github.com/vllm-project/vllm /tmp/vllm` 하고 읽은 파일:

| # | 파일 | 역할 |
|---|---|---|
| v1 | `csrc/attention/paged_attention_v1.cu` + `attention_kernels.cuh` | 오리지널 CUDA 커널 (2023). **Per-query-head 그리드.** |
| v2 | `csrc/attention/paged_attention_v2.cu` | ctx 축 split-k + reduce 커널. |
| triton | `vllm/v1/attention/ops/triton_unified_attention.py` | 현행 Triton 구현. **Per-KV-head 그리드.** |

### 발견 1 — 내 Phase 3.5 는 vLLM 의 현행 Triton 커널과 axis-for-axis 매치

| axis | vLLM Triton unified | 내 Phase 3.5 |
|---|---|---|
| grid | `(Σ q_blocks, H_kv)` | `(B, H_kv)` |
| Q tile | `(BLOCK_M, HEAD)` — 행 묶어 로드 | `(GROUP, HEAD)` |
| Matmul | `tl.dot(Q, K)` / `tl.dot(P, V)` | `tl.dot` (GROUP≥4) or manual fallback |
| Softmax | per-row fp32 running `(M, L, acc)` | 동일 |
| KV layout | `(num_blks, blk_size, H_kv, d)` | **동일** |
| BLOCK_SIZE | 가변 (block_size = V.shape[1]) | 8/16/32/64/128 |
| prefill | yes (BLOCK_Q 로 row 패킹) | no (decode only, 레슨 범위) |

vLLM 의 axis-0 가 `(batch × query block)` 이고 내 axis-0 가 pure batch 인 차이는, vLLM 은 prefill 까지 한 커널로 처리하니까 query 길이가 가변 — 그래서 query block 을 merge 해서 인덱싱. 나는 decode 만 하니까 `q_len=1` 로 고정, batch 가 바로 0축. **구조는 같음**, scope 가 다를 뿐.

KV layout 은 특히 재미있다. CUDA v1 은 `(NB, H_kv, d/x, BLK, x)` 로 복잡한 vectorization 레이아웃 (x=8 for fp16). Triton unified 는 `(NB, BLK, H_kv, d)` 로 **단순화**. 나는 논문 (x 에 대해 언급도 없는 simplification) 을 보고 단순한 레이아웃을 골랐는데, vLLM Triton 포트와 정확히 일치. 내가 2023 CUDA v1 을 베꼈으면 이 단순화를 놓쳤을 것.

### 발견 2 — vLLM 자신이 나와 같은 refactor 를 거쳤다

`paged_attention_v1.cu:86`:
```cpp
dim3 grid(num_heads, num_seqs, 1);
```

이것이 **per-query-head grid**. 내 Phase 3 의 디자인과 동일. 2023 년 vLLM 이 shipping 한 오리지널.

당시엔 왜 이게 괜찮았는가:
- LLM 시장의 대부분이 MHA (H_kv == H_q) → group redundancy 가 **구조적으로 없음**.
- 몇 안 되던 GQA 모델 (PaLM 의 MQA) 은 KV 가 작아서 L2 가 먹어줌.
- Triton 2.x 의 MMA 가 아직 CUDA/manual-vec 와 경쟁할 수 없어서 CUDA 가 정답.

LLaMA-2-chat, LLaMA-3, Mistral 이 GQA 로 shipping 되면서 per-query-head grid 가 병목. vLLM 은 Triton 으로 옮기면서 `(q_block, H_kv)` 로 restructure — **내가 Phase 3 → Phase 3.5 에서 한 refactor 와 정확히 같다**, 같은 forcing function (GQA group ≥ 4).

### 발견 3 — 내가 한 것 중 vLLM 이 안 한 것 하나

`triton_unified_attention.py:410`:
```python
S += scale * tl.dot(Q, K)      # input_precision 지정 없음
```

vLLM 은 `tl.dot` 에 precision 를 명시하지 않는다. **production 이 fp16/bf16 만 돌리니까 문제 없음**. 하지만 누가 fp32 로 그 path 를 돌리면 내가 본 것과 같은 4e-4 오차가 난다. 내 `IS_FP32` branching + `input_precision="ieee"` 는 "엄밀히 lesson context 에서만 중요한" 차이지만, 그래도 걸려냈다.

### 발견 4 — 내가 못 한 것: ctx 축 split-k

내 MQA 잔여 +85 % gap 의 원인. SDPA 는 이 shape 에서 698 GB/s (DRAM 의 2.3×) 를 낸다 — L2 가 반 이상 흡수. 1 KV head × 4k tokens × 128 dim × 2 B fp16 = 1 MB 가 L2 48 MB 에 쉽게 들어가고 32 query heads 가 공유해서 생기는 throughput.

내 paged 는 block_table indirection 때문에 같은 L2 reuse 가 안 된다. 구조적 한계. grid 만 바꿔서는 못 닫음.

vLLM 의 v2 (`paged_attention_v2.cu`) 와 `kernel_unified_attention_3d` 는 ctx 축을 (기본 512 토큰) partition 하고 reduce 커널로 softmax 재조합. 각 partition 안에선 L2 가 유지된다. 이걸 Phase 4.5 blueprint 로 남겨두고 Lesson 12 로 이월.

**교훈 #3**: **Paper + HW + workload 만으로 짜도 실전 소스와 수렴하면 그건 설계가 맞다는 증거**. 내가 vLLM 을 안 읽고 Phase 3.5 까지 완성했는데 vLLM 의 현행 Triton 포트와 axis-for-axis 매치. 이건 "내가 똑똑한 것" 이 아니라 "맞는 답이 하나" 라는 것. 이 convergence 를 기록하는 것 자체가 "설계가 맞다" 의 증거.

---

## 최종 숫자 (L4 sm_89, fp16, warmup=50, iters=200)

| shape | B | H | H_kv | group | SDPA ms | paged best (bs) | gap |
|---|---|---|---|---|---|---|---|
| llama7b MHA | 8 | 32 | 32 | 1 | 1.322 | 1.227 (bs=16) | **-7 %** |
| llama7b MHA | 8 | 32 | 32 | 1 | 6.115 ctx=8k | 4.927 (bs=64) | **-19 %** |
| **llama38b GQA** | **8** | **32** | **8** | **4** | **0.308** | **0.264 (bs=16)** | **-14 %** |
| llama38b GQA | 32 | 32 | 8 | 4 | 1.163 | 1.197 (bs=128) | +3 % |
| llama70b GQA | 4 | 64 | 8 | 8 | 0.049 | 0.048 (bs=128) | **-1 %** |
| llama70b GQA | 8 | 64 | 8 | 8 | 0.532 | 0.526 (bs=16) | **-1 %** |
| mqa | 16 | 32 | 1 | 32 | 0.048 | 0.089 (bs=128) | +85 % |

Correctness: 32/32 PASS, fp16 max diff ≤ 1e-3, fp32 max diff ≤ 4e-7.

LLaMA-3-8B B=8 ctx=2k 에서 **SDPA (= Tri Dao FA-2 CUDA) 를 14 % 이기는** 275 줄 Triton 커널. LLaMA-70B 는 parity. MQA 는 +85 % (split-k 로 닫힐 residual).

---

## 세 가지 남는 것

### (1) Correctness 가 통과하는 것이 "맞다" 를 의미하지 않는다

32/32 PASS 직후 "끝났다" 고 판단했으면 GQA shape 에서 **2-13 배 느린** 커널을 ship 했다. 이 함정은 **속도 벤치에 SDPA gap 컬럼** 이 없었으면 못 잡는다. `allclose` + gap 을 **함께** 리포트. 이후로는 새 커널에 이 두 줄이 default.

### (2) 버그가 버그 뒤에 숨는다

Grid bug (Phase 3) 와 TF32 bug (Phase 3.5) 는 독립적이었고 **순차적으로만 드러났다**. Grid 를 안 고쳤으면 TF32 가 안 나타남. 고치자마자 나타남. 큰 refactor 뒤엔 correctness 를 **반드시** 재실행 — "한 버그 고쳤으니 안전" 은 정확히 저 상황에서 틀린다.

### (3) 독립적으로 수렴하는 게 설계의 증거

vLLM 소스를 Phase 3.5 끝난 뒤에 읽었는데 axis-for-axis 매치. **이건 내가 똑똑한 게 아니라, 맞는 답이 하나고 같은 툴 (Triton) + 같은 HW (sm_80+) + 같은 workload (GQA) 면 거기로 수렴한다** 는 뜻. 오히려 이 convergence 를 명시적으로 기록하는 게 credible — "ecosystem 이 이미 한 refactor 를 miniature 로 재현했다" 는 스토리.

반대로 내가 vLLM 안 한 것 (IEEE 강제) 은 context 차이 — production 은 fp16/bf16 이라 필요 없고, lesson 은 fp32 correctness 도 테스트하니까 필요. **공통점과 차이점을 둘 다 명확히 기록** 하면 설계 판단이 말이 된다.

---

## 다음 세션 — 남은 +85 % 를 닫으려면

**Phase 4.5-b (Lesson 12 입구 후보)**: ctx 축 split-k.

```
grid = (B, H_kv, SEGMENTS)           # axis 2 가 새로 추가
SEGMENTS = ceil(ctx / PARTITION_SIZE)  # e.g. 512-token partition
```

각 program 은 `PARTITION_SIZE` 토큰의 partial softmax 계산 (`max`, `expsum`, `out`) 후 scratch 에 저장. 두 번째 reduce 커널이 `(B, H_kv)` 위에서 partition 들의 max-logit 로 normalize 해서 recombine. 이게 정확히 vLLM 의 v2.

L4 기준 예상:
- MQA B=16 ctx=4k: 16 programs → ~128 programs. 58 SM 위에서 2.2 wave, SDPA 와 경쟁.
- 두 kernel launch 오버헤드 +15 µs. 짧은 ctx 에서는 regression 이니 `PARTITION_SIZE` conditional.

구현 추정: **반나절**. 쉬운 길에 있다 — 이 글 쓰는 시점에도 커널 안의 블록 루프를 segment 축으로 쪼개기만 하면 되니까. Lesson 12 의 `prefill + decode 통합` 과 엮으면 vLLM v1 수준 재현에 가깝다.

---

## 재현용 커맨드

```bash
# 1회: VM 생성
./scripts/gcp_create_l4_spot_vm.sh <PROJECT_ID> us-west1-b cuda-l4-dev-lesson11

# Phase 1+2 · correctness (32 shape)
python3 triton_kernels/bench/bench_paged_attention.py

# Phase 3+3.5 · speed bench (SDPA vs paged, 10 shapes × 5 block sizes)
python3 triton_kernels/bench/bench_paged_attention_speed.py
```

GCP DL image 기본 설정은 lesson 10 handoff 와 동일 — ncu 는 `sudo -E env PATH=$PATH ncu ...` 로 감싸면 profiling permission 우회 가능.

### 재료 링크

- Lesson 11 핸드오프: [`docs/lesson_11_handoff_2026-04-22.md`](/Users/xavier/dev/cudatraining/docs/lesson_11_handoff_2026-04-22.md:1)
- Phase 3 + 3.5 상세: [`docs/lesson_11_phase3_findings.md`](/Users/xavier/dev/cudatraining/docs/lesson_11_phase3_findings.md:1)
- Phase 4 (vLLM 소스 diff + split-k blueprint): [`docs/lesson_11_phase4_vllm_diff.md`](/Users/xavier/dev/cudatraining/docs/lesson_11_phase4_vllm_diff.md:1)
- Lesson 09 (전편, contiguous MHA): [`docs/blog_draft_lesson_09_mha_causal_fa.md`](/Users/xavier/dev/cudatraining/docs/blog_draft_lesson_09_mha_causal_fa.md:1)
- Lesson 10 (프로파일링): [`docs/blog_draft_lesson_10_profiling.md`](/Users/xavier/dev/cudatraining/docs/blog_draft_lesson_10_profiling.md:1)
- vLLM 논문: Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023.
- 내가 읽은 vLLM 소스 (HEAD `2463f00`):
  - `csrc/attention/paged_attention_v1.cu` — 오리지널 CUDA 런처
  - `csrc/attention/paged_attention_v2.cu` — split-k 버전
  - `csrc/attention/attention_kernels.cuh` — 실제 kernel body
  - `vllm/v1/attention/ops/triton_unified_attention.py` — 현행 Triton replacement
