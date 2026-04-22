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

## Day 2 target (다음 세션)

Day 1 의 질문 1-3 해결:

- `selector.py` 165 줄 읽어서 dispatch rule 정리.
- Triton 3 파일 (decode / prefill / unified) 각각의 kernel signature + 언제 호출되는지 확인.
- `triton_merge_attn_states.py` 읽어서 **우리 lesson 12 reduce 와 구조 비교** — 같은가? 다른가? 다르면 어디가?

**산출물**: 이 문서의 Day 2 섹션 추가 + 3 개의 Triton Python 파일을 **각각 50 줄씩 요약한 cheat sheet**.

Week 1 전체 산출물은 이 문서의 §Day 5 에 끝나고, §Candidate 3 에서 Week 2 에 팔 한 개를 선정한다.

---

## 재료

- vLLM HEAD `a490513` · `~/dev/vllm` · shallow clone
- Lesson 11 blog (vLLM v0 시대 관찰): [`docs/blog_draft_lesson_11_paged_attention.md`](/Users/xavier/dev/cudatraining/docs/blog_draft_lesson_11_paged_attention.md:1)
- Lesson 12 blog (split-k + auto-dispatch): [`docs/blog_draft_lesson_12_split_k.md`](/Users/xavier/dev/cudatraining/docs/blog_draft_lesson_12_split_k.md:1)
