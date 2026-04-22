# vLLM Audit 01 — Attention Subsystem Path Map

Lesson 13, Week 1. Read-only exploration. 목표: vLLM 의 attention subsystem 을 end-to-end 로 매핑 후 레슨 12 개선점을 꽂을 자리 3 개 도출.

Clone 정보: `~/dev/vllm`, HEAD `a490513` (shallow depth=1). 이 문서는 Day 1-7 에 걸쳐 자라는 living doc.

---

## Day 1 — Surface scan (completed)

### 큰 발견: **v0 는 완전히 제거됐음**

레슨 11 때만 해도 `vllm/attention/` (v0) 와 `vllm/v1/attention/` 가 공존했는데, 현재 HEAD 에는 v0 디렉터리 자체가 존재하지 않음. vLLM 은 이제 **v1-only**. 레슨 11/12 의 context 가 일부 outdated — 그래서 이 audit 가 필요하다.

### 3-layer 구조

```
vllm/v1/attention/
├── selector.py              (165 lines)  ← backend dispatch
├── backend.py               (1000+ lines) ← ABC + AttentionMetadata
├── backends/                              ← 18+ backends (HW × variant)
│   ├── triton_attn.py       (771 lines)
│   ├── flash_attn.py
│   ├── flashinfer.py
│   ├── flex_attention.py
│   ├── tree_attn.py                       (speculative decoding)
│   ├── flash_attn_diffkv.py               (diff KV per head)
│   ├── turboquant_attn.py                 (quantized)
│   ├── mamba{1,2}_attn.py, gdn_attn.py    (state-space / linear attn)
│   ├── short_conv_attn.py
│   ├── cpu_attn.py, rocm_*.py, xpu_*.py   (HW-specific)
│   ├── mla/                               (DeepSeek MLA: 13 variants)
│   │   ├── triton_mla.py, cutlass_mla.py
│   │   ├── flashattn_mla.py, flashinfer_mla.py, flashmla.py
│   │   ├── *_sparse.py (5개)
│   │   └── ...
│   └── registry.py, utils.py, fa_utils.py
└── ops/                                   ← 실제 kernel 구현
    ├── triton_unified_attention.py        ← lesson 11 에서 봤던 그것
    ├── triton_decode_attention.py         ← decode 전용
    ├── triton_prefill_attention.py        ← prefill 전용
    ├── triton_merge_attn_states.py        ← split-k reduce 커널
    ├── triton_reshape_and_cache_flash.py
    ├── chunked_prefill_paged_decode.py    ← mixed batch
    ├── prefix_prefill.py                  ← prefix caching
    ├── paged_attn.py                      ← 공용 paged
    ├── triton_turboquant_{store,decode}.py
    ├── flashmla.py
    └── common.py, merge_attn_states.py
```

### CUDA 커널 (`csrc/attention/`)

```
csrc/attention/
├── paged_attention_v1.cu                  ← 레슨 11/12 에서 참조
├── paged_attention_v2.cu                  ← split-k v2
├── attention_kernels.cuh                  ← 공용 kernel body
├── attention_utils.cuh, attention_generic.cuh
├── dtype_{float16,bfloat16,float32,fp8}.cuh
├── merge_attn_states.cu                   ← reduce 커널 (CUDA 판)
├── vertical_slash_index.cu                ← sparse attention index
└── mla/                                   ← MLA 전용 CUDA
```

### backend.py 의 key class 들

- `AttentionType(Enum)` — DECODER / ENCODER / ... 구분
- `AttentionBackend(ABC)` — backend 인터페이스 (line 55)
- `AttentionMetadata` — runtime state (line 338)
- `CommonAttentionMetadata` — backend-agnostic state (line 346)
- `AttentionCGSupport(Enum)` — CUDA graph 지원 단계 구분 (line 481)
- `AttentionMetadataBuilder(ABC)` — metadata 빌드 (line 498)
- `AttentionImpl(ABC)` — forward pass 구현 인터페이스 (line 745)
- `MLAAttentionImpl`, `SparseMLAAttentionImpl` — DeepSeek MLA 전용

### selector.py 의 dispatch 함수

- `get_attn_backend(...)` (line 50) — 공개 API
- `_cached_get_attn_backend(...)` (line 108) — 캐시 래퍼
- `get_mamba_attn_backend(...)` — state-space 분기

→ 이 함수가 (HW, dtype, head_dim, model config, ...) 을 보고 어느 backend 를 return 하는지가 Day 3-4 의 타깃.

---

## Day 1 에 생긴 질문들 (Day 2+ 숙제)

1. **왜 Triton attention 이 3 개 파일로 분리돼 있나?**
   - `triton_decode_attention.py`
   - `triton_prefill_attention.py`
   - `triton_unified_attention.py`
   - 레슨 11 에선 unified 한 개만 봤는데, 지금은 decode/prefill 각각 따로 존재 + unified 도 여전히 존재. 셋이 다 살아 있는 이유? 언제 각각 쓰이나?

2. **`triton_merge_attn_states.py` 가 우리 레슨 12 reduce 커널과 같은 역할인가?** 파일 이름 매치 — vLLM 도 split-k 로 online softmax recombination 하는 reduce 커널을 분리한 형태. **이게 우리 auto-dispatch 를 꽂을 자리일 가능성이 높음**.

3. **Backend selector 의 dispatch rule 은?** `selector.py` 가 shape/dtype/HW 로 어떻게 라우팅? L4 (sm_89, fp16) + LLaMA-3-8B 에서 어떤 backend 가 선택되나?

4. **`chunked_prefill_paged_decode.py` 가 혼합 batch 처리** — 이게 바로 내가 블로그 section 10 에서 "unified kernel" 후보로 언급했던 그것. **실제로 어떻게 구현됐는지 보면 "unified kernel" 에 PR 낼 거리가 있는지 판단 가능**.

5. **MLA 가 왜 이렇게 많나 (13 개 variants)?** DeepSeek 이 핫해서인지, 각 HW/precision 조합별로 분리된 건지. 레슨 범위 밖이지만 mapping 에는 포함.

6. **v0 제거 시점**: git log 가 shallow 라 정확히 언제 사라졌는지는 나중에 `fetch --unshallow` 하면 확인 가능.

---

---

## Day 2 — Triton kernels + selector + 두 개의 오해 수정

### 오해 1 — `triton_merge_attn_states.py` 는 split-k reduce 가 **아님**

Day 1 추측: "이게 우리 lesson 12 reduce 와 같은 역할일 것 같다." → **틀림**.

읽어보니:
- 정확히 **2-way merge** (prefix + suffix). N 개 segment 를 recombine 하는 게 아님.
- arxiv:**2501.01005 §2.2** 를 구현 (파라미터 이름이 `prefix_lse`, `suffix_lse`).
- 용도: **prefix caching**. 캐시에서 가져온 prefix 의 partial attention + 새 suffix 의 attention 을 합칠 때.
- Grid `(num_tokens, num_query_heads)`, 토큰 단위 element-wise merge.
- `prefill_tokens_with_context` boundary 로 **mixed batch** (context 있는 토큰 + 없는 토큰) 처리.

완전히 다른 문제. 우리 split-k reduce 와는 구조도 용도도 다름.

### 오해 2 — 우리가 찾던 split-k reduce 는 별도 파일이 아니라 unified 안에 있음

실제 split-k reduce 는 `triton_unified_attention.py:926` 의 **`reduce_segments` kernel** (파일 안에 inline). 즉 vLLM 은 split-k forward + reduce 를 **한 파일에 묶어놨음**. 우리는 lesson 12 에서 `paged_attention.py` 에 두 kernel 을 나란히 둠 — 같은 설계.

### `selector.py` 165 줄 읽기 결과

얇은 wrapper. 실제 dispatch logic 은 두 겹 아래:

```
selector.get_attn_backend(head_size, dtype, ...)
  → _cached_get_attn_backend(backend, config)
    → current_platform.get_attn_backend_cls(backend, config)   ← HW-specific
       → vllm/platforms/cuda.py   (NVIDIA 분기)
```

즉 **L4 sm_89 + fp16 + LLaMA-3-8B 에서 어떤 backend 가 선택되나** 는 `vllm/platforms/cuda.py` 를 읽어야 함. Day 3 타깃.

### Triton 3 파일의 역할 분리 (Day 1 의 질문 1 답)

**정리 표**:

| 파일 | lines | grid | paged? | split-k? | GQA | 용도 |
|---|---|---|---|---|---|---|
| `triton_unified_attention.py` | 1315 | 2D: `(Q_blocks, H_kv)` <br>3D: `(Q_blocks, H_kv, 16)` | ✅ | ✅ (16 segments hardcoded, inline reduce) | ✅ `BLOCK_M = max(16, pow2(q_per_kv))` | **메인** — prefill + decode paged |
| `triton_decode_attention.py` | 778 | stage1 `(B, H, NUM_SPLITS)` + stage2 `(B, H)` | ❌ (contiguous) | ✅ (always, no heuristic) | ✅ (별도 grouped 변형) | decode-only, contiguous |
| `triton_prefill_attention.py` | 253 | `(B, H, M_blocks)` | ❌ | ❌ | ✅ `kv_group_num` | 초기 prefill (캐시 없을 때), SGLang adapt |
| `triton_merge_attn_states.py` | 175 | `(num_tokens, H_q)` | n/a | n/a (2-way merge) | n/a | **prefix caching**, 완전 다른 문제 |

### 우리 lesson 12 vs vLLM unified 의 실제 diff

| 측면 | 우리 (lesson 12) | vLLM unified |
|---|---|---|
| split-k 결정 | `B*H_kv < 0.5*SM ∧ segments ≥ 4` — **SM utilization 중심** | `num_seqs > 128/num_kv_heads` — **batch-count 중심** |
| SEGMENTS | `ceil(ctx / PARTITION_SIZE)` — **ctx 적응형** | **16 hardcoded** — 짧은 ctx 에선 낭비, 긴 ctx 에선 부족 |
| Reduce kernel | 별도 파일 (`paged_attention.py` 안 나란히) | 같은 파일 inline (`reduce_segments`) |
| Prefill 통합 | ❌ (decode only) | ✅ (한 kernel) |
| Scratch allocation | 호출 시 alloc | **backend 에서 사전 alloc** (`softmax_segm_output` 등 `seq_threshold_3D × H_q × 16 × padded_d`) — warm path 빠름 |
| Precision (fp32) | `input_precision="ieee"` 명시 | 명시 없음 (fp16/bf16 위주 가정) |

### Day 2 에 새로 드러난 candidate 후보

**Candidate A — adaptive NUM_SEGMENTS for vLLM unified**
- vLLM 이 16 segments 하드코딩 → 긴 ctx 에선 segment 당 작업이 너무 커서 SM 점유율 회복이 부족. 짧은 ctx 에선 낭비.
- 우리 lesson 12 의 `ceil(ctx / PARTITION_SIZE)` 로직 + auto-dispatch heuristic 을 unified kernel 의 dispatcher 에 포팅.
- 난이도: **낮음-중간**. Dispatch wrapper 쪽 Python 만 건드리면 됨. Triton kernel 본체 변경 없음.
- 임팩트: **불확실** — 16 이 "good enough" 일 수도. 벤치 필요 (Day 5).

**Candidate B — SM-utilization based heuristic**
- 현재 vLLM 은 `num_seqs > 128/num_kv_heads` — 이건 "H_kv 가 클 때만 batch 커지면 split-k 안 씀" 뜻. 우리 lesson 12 에서 봤듯, 진짜 문제는 `B*H_kv` 가 SM 대비 작을 때. 우리 heuristic 이 직접 문제 정조준.
- 난이도: **중간**. Dispatch 로직 + 테스트 케이스 업데이트.
- 임팩트: **높음** — MQA 같은 edge shape 에서 이미 vLLM 이 잘못 고르고 있을 수 있음 (Day 5 에서 확인).

**Candidate C — prefix-caching merge 를 N-way 로 일반화**
- 현재 `merge_attn_states` 는 2-way (prefix + suffix). N-way 면 중첩 prefix (A → AB → ABC) 를 한 번에 merge 가능. 현재는 반복 호출.
- 난이도: **높음**. 수학은 간단하지만 API breaking change 가능성 + prefix caching 구조 이해 필요.
- 임팩트: **중간** — prefix caching 은 hot 이지만 2-way 중첩의 실제 빈도 알려면 측정 필요.

**Candidate D — fp32 IEEE path 추가**
- vLLM unified 가 `input_precision` 명시 안 함. production 은 fp16/bf16 이라 문제 없지만, research / debug 에선 정밀도 이슈. 우리 lesson 11 에서 정확히 겪은 건.
- 난이도: **낮음**. 분기 한 줄 + 테스트.
- 임팩트: **낮음-중간**. Niche 지만 실제 요청하는 사용자 있음.

---

## Day 3 — Candidate B 로 직행 피봇 (in progress)

Day 1-2 결과를 다시 읽었을 때 판단: 원래 Day 3-5 의 "platforms/cuda.py dispatch rule 완전히 드러내기 + end-to-end request trace" 는 Week 1 전체를 잡아먹는데, 거기서 나오는 지식은 **candidate 선정** 에 필요하지 **검증** 에는 덜 필요하다. Candidate 4 개 중 B 는 이미 근거(Day 2, lesson 12 bench numbers) 충분히 강함 → Week 2 의 검증 단계로 바로 진입하는 편이 ROI 높음.

**새 Week 1/2 의 re-scoping:**

| 기존 plan | 신 plan |
|---|---|
| Day 3: platforms/cuda.py 읽기 | Day 3-5: Candidate B 의 Stage 1 — kernel-level bench 준비/실행 |
| Day 4: TritonAttentionBackend trace | (나중에 필요하면 Week 3 에 복귀) |
| Day 5: lesson 12 vs vLLM 직접 bench | (Candidate B Stage 1 에 포함) |
| Day 6-7: candidate 최종 선정 | Week 2 Day 1-2: bench 결과 분석 및 문서화 |

### Candidate B 논리 요약 (재확인용)

- vLLM `vllm/v1/attention/backends/triton_attn.py:163` 이 dispatch threshold 를 결정:
  ```python
  seq_threshold_3D = 128 // num_kv_heads
  ```
  → `ops/triton_unified_attention.py:1157-1166` 의 `num_seqs > seq_threshold_3D` 분기에서 2D(single-pass) vs 3D(split-k) 결정.
  → 동치: `B * H_kv > 128 → single-pass`, `B * H_kv ≤ 128 → split-k`.
- 이 `128` 은 magic number. SM 수에 따라 scale 되어야 할 값인데 HW-agnostic 고정상수.
- GPU 별 SM 수:
  - L4: **58 SMs** (우리 타겟)
  - A100: 108 SMs
  - H100: 132 SMs (← 128 이 여기서 맞음)
  - B200: 148 SMs
- L4 에서 `B*H_kv ∈ [30, 128]` 구간은 vLLM 이 split-k 로 가는데, 우리 lesson 12 측정으로는 single-pass 가 빠름 (reduce kernel launch overhead 가 splitting gain 을 상쇄).

### Candidate B 의 Stage 전략

**Stage 1 (Week 2 핵심, 우선)** — isolated kernel-level bench
- 근거: "kernel+dispatch 둘을 다 바꾸면 어디서 얻었는지 모호". 먼저 kernel 은 vLLM 의 것을 그대로 쓰고 dispatch threshold 만 변형해 gap 을 측정.
- 산출물:
  - `triton_kernels/vllm_extracted/unified_attention.py` — vLLM unified kernel 을 min-deps extraction (NOTICE.md 와 함께)
  - `triton_kernels/bench/bench_vllm_vs_ours.py` — 3-way 벤치 (ours / vllm-default-threshold / vllm-with-SM-aware-threshold)
  - `docs/vllm_audit_02_heuristic_b_kernel_bench.md` — 결과 문서
- 3-way 의 설계 이유:
  - `vllm-sm-aware` 는 vLLM 의 kernel 을 그대로 쓰되 threshold 만 `(num_SMs // 2) // H_kv` 로 교체. 만약 `vllm-sm-aware ≈ ours < vllm-default` 이면 "**win 은 dispatch 에서 왔다, kernel quality 아니다**" 가 깔끔하게 증명됨. PR 논리의 핵심.

**Stage 2 (Week 3, Stage 1 이 pass 하면)** — vLLM e2e 벤치
- 실제 vLLM 서버를 세워서 LLaMA-3-8B 같은 모델로 throughput/latency 측정. Stage 1 의 kernel-level 숫자가 e2e 에서도 유효한가.
- Stage 1 이 negative/marginal 이면 Stage 2 는 skip.

**Stage 3 (Week 4, Stage 1+2 둘 다 pass 하면)** — PR 준비
- Upstream PR 이슈/디스커션 먼저 열어서 maintainer 반응 본 후 submit.

### Day 3 진행 상황 (현재 세션)

- [x] Stage 1 Day 1: vLLM `triton_unified_attention.py` (1315 lines) 추출 → `triton_kernels/vllm_extracted/unified_attention.py`
  - 5 개 import 만 교체 (envs, logger, current_platform, triton_utils, KVQuantMode); kernel body 는 byte-identical
  - `is_batch_invariant` 분기는 `= False` constant 로 유지하여 upstream diff 보존
  - `NOTICE.md` 에 변경 내역 + Apache 2.0 attribution 기록
  - AST syntax check: 1345 lines, no live `vllm.` references outside comments
- [x] Stage 1 Day 1-2: 3-way bench harness → `triton_kernels/bench/bench_vllm_vs_ours.py`
  - **correctness phase**: smoke test (B=2, H_q=8, H_kv=4, ctx=768, fp16) — ours-SP/SK + vllm-2D/3D + PyTorch ref 6-way allclose
  - **dispatch-heuristic phase**: 7 shapes 커버
    - vLLM 이 틀릴 것으로 예상: LLaMA-7B MHA B=1 ctx=1k/4k, LLaMA-70B GQA B=4 ctx=2k/4k (B×H_kv=32, < 128 but ≥ 29)
    - 둘 다 SK: MQA B=8 H_kv=1 ctx=4k
    - 둘 다 SP: LLaMA-7B B=32 ctx=1k, LLaMA-7B B=1 ctx=256
  - 각 shape 마다 경로(SP/SK | 2D/3D) 와 시간, ratio 출력. Summary 섹션에서 `vllm-default > ours +5%` 이면서 `vllm-smaware ≈ ours` 인 케이스를 자동 flag.

### Stage 1 Day 3-5 진행 예정

- [ ] Day 3: GCP L4 VM 재가동 → `bench_vllm_vs_ours.py` 실행 → 결과 JSON/CSV 로 저장
- [ ] Day 4: 결과 분석 — vLLM-default 가 실제로 예상한 shape 에서 regression 보이는가? vllm-sm-aware 가 ours 와 정말 같은가?
- [ ] Day 5: `docs/vllm_audit_02_heuristic_b_kernel_bench.md` 작성 (methodology, raw numbers, plots, 결론)

### Day 3 의 오픈 질문

1. **Stage 1 bench shape 매트릭스** — 7 개로 충분? 아니면 더 dense sweep (B ∈ {1,2,4,8,16,32}, H_kv ∈ {1,2,4,8,16,32}) 을 해서 heatmap 을 뽑을까? 전자 먼저, 결과 보고 dense 여부 결정.
2. **L4 외 GPU 에서의 cross-check** — Stage 1 에서 L4 만 쓴다면 "L4 만의 문제 아니냐" 반론 여지. A100 도 있으면 이상적이지만 비용 부담. 우선 L4 에서 완결하고 PR 이슈 열 때 "L4 에서 관측, A100/H100 에서도 같은 논리로 miscalibrated 일 가능성 있음, 유저가 재현 가능" 으로 프레임.
3. **Stage 2 로 넘어갈 기준선** — Stage 1 결과가 어느 정도여야 Stage 2 로 승격? 초안: "L4 에서 realistic shape 3 개 이상에 +10% 이상 latency 차이" (lesson 12 가 측정한 수치 중 +19% / +38% 는 이 기준 통과).

---

---

## 재료

- vLLM HEAD `a490513` · `~/dev/vllm` · shallow clone
- Lesson 11 blog (vLLM v0 시대 관찰): [`docs/blog_draft_lesson_11_paged_attention.md`](/Users/xavier/dev/cudatraining/docs/blog_draft_lesson_11_paged_attention.md:1)
- Lesson 12 blog (split-k + auto-dispatch): [`docs/blog_draft_lesson_12_split_k.md`](/Users/xavier/dev/cudatraining/docs/blog_draft_lesson_12_split_k.md:1)
- Candidate B Stage 1 산출물:
  - 추출된 kernel: [`triton_kernels/vllm_extracted/unified_attention.py`](/Users/xavier/dev/cudatraining/triton_kernels/vllm_extracted/unified_attention.py:1) · [`NOTICE.md`](/Users/xavier/dev/cudatraining/triton_kernels/vllm_extracted/NOTICE.md:1)
  - bench harness: [`triton_kernels/bench/bench_vllm_vs_ours.py`](/Users/xavier/dev/cudatraining/triton_kernels/bench/bench_vllm_vs_ours.py:1)
