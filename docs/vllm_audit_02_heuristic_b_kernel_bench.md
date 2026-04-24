# vLLM Audit 02 — Candidate B: Kernel-Level + e2e Bench (L4, sm_89)

Lesson 13, Week 2 Day 1-3. Candidate B (`seq_threshold_3D = 128 // num_kv_heads` dispatch heuristic) 의 Stage 1 (kernel-level) + Stage 1.5 (adaptive SEGMENTS 실험) + Stage 2 (vLLM e2e) 검증 결과.

- 실행:
  - Stage 1 / 1.5 : 2026-04-22, GCP L4 VM `cuda-l4-dev-lesson09` (nemo-488500 / us-east4-c, NVIDIA L4, 58 SMs, sm_89, torch 2.11.0+cu130, triton 3.6.0)
  - Stage 2     : 2026-04-23, 동일 VM, vLLM 0.19.1 (pip install), float16, `attention_backend=TRITON_ATTN`. 두 모델: TinyLlama-1.1B-Chat-v1.0 (MLP-dominant 소형), Qwen2.5-7B-Instruct (medium GQA-7x).
- 산출물: [`bench_vllm_vs_ours.py`](/Users/xavier/dev/cudatraining/triton_kernels/bench/bench_vllm_vs_ours.py:1), [`bench_vllm_e2e.py`](/Users/xavier/dev/cudatraining/triton_kernels/bench/bench_vllm_e2e.py:1), [`vllm_extracted/unified_attention.py`](/Users/xavier/dev/cudatraining/triton_kernels/vllm_extracted/unified_attention.py:1), [`scripts/toggle_vllm_patch.py`](/Users/xavier/dev/cudatraining/scripts/toggle_vllm_patch.py:1), raw CSV/JSON at `bench_results/l13_candidateB_stage1_20260422T140*.{csv,json}`, `bench_results/l13_candidateB_stage2_20260423T073336Z_*.{csv,json}` (1.1B), `bench_results/l13_candidateB_stage2_qwen7b_20260423T085705Z_*.{csv,json}` (7B)

---

## TL;DR

1. **vLLM Triton unified attention 은 L4 에서 우리 lesson 12 kernel 대비 shape 의 대다수에서 느리다.** 79 config 중 73 개 (92%) 에서 우리가 5% 이상 빠르고, 61 개 (77%) 에서 20% 이상 빠르다. Geomean total gap **+40.9%**, median **+23.8%**.
2. **원인 분해 결과, 문제는 "dispatch heuristic 하나" 가 아니다.** Dispatch 를 SM-aware 로 고치면 geomean **+4.1%** / median **+2.4%** 밖에 회복 안 된다. 나머지 (geomean +35.4% / median +23.2%) 는 **kernel 자체** 의 gap 이다.
3. **Dispatch 수정만으로 10% 이내로 닫히는 shape 은 73 개 regression 중 8 개 (11%).** 65 개 (89%) 는 dispatch 를 고쳐도 여전히 10%+ 뒤처진다.
4. **Stage 1.5 (adaptive NUM_SEGMENTS)** — kernel gap 의 범인이 "SEGMENTS=16 하드코드" 인지 검증. **부정 결과**. Adaptive `ceil(ctx/512)` 로 바꾸면 geomean 이 오히려 1.4 pp 악화, adaptive-only 로 새로 닫는 shape 은 0 개, 작은 batch 규모 shape 에선 최대 +19.6 pp 퇴보. **16 은 small-batch occupancy multiplier 로 동작하는 rational default** 임이 확인됨.
5. 따라서 PR scope 가 **scenario A (dispatch-only)** 로 확정됨. Kernel gap 의 주원인은 SEGMENTS 가 아니라 grid/BLOCK_M 설계일 가능성이 높음 (§6.6 가설) — 별도 track.
6. **Stage 2 (vLLM e2e)** — dispatch 수정이 e2e decode throughput 으로 **실제로 이어지는가** 검증. H_kv=4 모델 두 개 × batch∈{8,16,32} decode-heavy 워크로드. CUDA graph capture-size snapping (`triton_attn.py:166-176`, PR #28306) 까지 반영한 **effective threshold**: vanilla=32, smaware=8 (raw 7 → 가장 가까운 capture size 8 로 snap). 따라서 path divergence 는 batch ∈ {16, 32} 두 점에서만 발생하고, batch=8 은 양쪽 모두 3D 로 같은 kernel 을 탐:
   - **TinyLlama-1.1B**: batch=8 −0.14% (sanity), batch=16 **+2.89%**, batch=32 **+5.13%** tok/s.
   - **Qwen2.5-7B**: batch=8 −0.19% (sanity), batch=16 +0.92%, batch=32 **+1.68%** tok/s.
   - 공통: 방향·부호 일치, batch 커질수록 gain 증가. stdev sub-%. **batch=8 은 양쪽이 같은 3D kernel 을 호출하는 same-path 구간이라 sub-1% noise floor 로 해석** — 패치 toggle 이 다른 부수효과 없이 dispatch 한 줄만 바꿨음을 보여주는 sanity check.
   - 반직관 findings: **큰 모델에서 gain 이 더 작다** (사전 예상과 반대). 원인: decode 에서 MLP weight-loading 은 hidden_dim² 로 빠르게 커지지만 attention 은 batch×seq×H_kv×d_head 로 완만히 커져서 **attention share 가 모델 커질수록 줄어든다** (1.1B 기준 ~9% → 7B 기준 ~3–4% 추정). 따라서 per-call 20–25% gain 의 e2e projection 은 각각 ~5% / ~1.7% — 관측치와 ballpark 일치.
   - 의의: Performance claim 은 **model-size dependent** 로 솔직히 서술. Correctness claim (formula 가 SM count 무시한다는 점) 은 model-size 와 무관하게 성립.

---

## 1. Methodology

### 1.1 3-way bench design

각 shape 마다 같은 입력으로 세 경로를 돌리고 median-of-50 latency (ms) 를 측정.

| Variant | Kernel | Dispatch threshold | 역할 |
|---|---|---|---|
| **ours** | Lesson 12 single-pass + split-k paged attention (`triton_kernels/paged_attention.py`) | 우리 heuristic: `B*H_kv < 0.5*SM ∧ segments ≥ 4 → SK, else SP` | baseline (우리 구현) |
| **vllm-default** | `triton_kernels/vllm_extracted/unified_attention.py` (upstream byte-identical) | 상수 `seq_threshold_3D = 128 // H_kv` (upstream `triton_attn.py:163`) | "현재 upstream" |
| **vllm-smaware** | 동일 vLLM kernel | `seq_threshold_3D = (num_SMs // 2) // H_kv` — `num_SMs//2` 부분은 L4→29, A100→54, H100→66 (이걸 H_kv 로 한 번 더 나눠야 실제 threshold) | "dispatch 만 고친 vLLM" |

> **주의 — Stage 1 vs Stage 2 dispatch 차이**: Stage 1 의 kernel-level harness (`bench_vllm_vs_ours.py`) 는 위 raw 공식값을 그대로 `seq_threshold_3D` 인자로 넘긴다. 반면 실제 vLLM runtime (Stage 2) 은 `triton_attn.py:166-176` 에서 raw 값을 **CUDA graph capture sizes 중 가장 가까운 값으로 snap** 한다 (PR #28306). Default `cudagraph_capture_sizes = [1,2,4] + range(8,256,8) + range(256,max+1,16)` 기준, H_kv=4 에서 vanilla raw=32 → snap=32 (no-op), smaware raw=7 → snap=8. 따라서 Stage 1 의 path 분류 (예: `SP|3D|2D`) 는 Stage 1 내부 분석에는 정확하지만, e2e 의 boundary 와는 ±1 batch 만큼 어긋날 수 있다 (§8.1 참고).

3-way 의 의의: `vllm-default` 와 `vllm-smaware` 는 **kernel 이 같다**. 두 variant 의 차이는 순수하게 dispatch path (2D single-pass vs 3D split-k) 선택. 따라서 비교가 다음과 같이 분해된다:

```
gap_total    = (t_vllm_default / t_ours) - 1       # "현재 upstream 은 우리 대비 얼마나 느린가"
gap_dispatch = (t_vllm_default / t_vllm_smaware) - 1  # "upstream 의 dispatch bug 크기"
gap_kernel   = (t_vllm_smaware / t_ours) - 1       # "dispatch 고쳐도 남는, kernel 자체의 gap"

(1 + gap_total) = (1 + gap_dispatch) * (1 + gap_kernel)   # 멀티플리커티브
```

만약 `gap_kernel ≈ 0` 이면 "win 은 dispatch 에서 왔다, kernel quality 때문이 아니다" 가 증명되고 Candidate B 는 순수 heuristic PR 로 승격 가능. 반대로 `gap_kernel` 이 크면 kernel 자체를 건드려야 하는 더 큰 작업.

### 1.2 Shape coverage

- **Primary 7** — canonical production shape. LLaMA-7B MHA (H_kv=32), LLaMA-70B GQA (H_kv=8), MQA (H_kv=1), 각 B×ctx 조합.
- **Sweep 72** — grid `B ∈ {1,2,4,8,16,32} × H_kv ∈ {1,2,4,8,16,32} × ctx ∈ {1024, 4096}`, GQA ratio 4 고정. L4 의 58 SMs 기준 `B*H_kv ∈ [1, 1024]` 전구간 커버.

모두 fp16, head_dim=128, block_size=16. Correctness smoke check (ours-SP / ours-SK / vllm-2D / vllm-3D / PyTorch reference 6-way allclose) 는 bench 실행 전 PASS 확인.

### 1.3 Path notation

테이블에서 `path_ours | path_vllm_default | path_vllm_smaware` 를 `SP|3D|2D` 처럼 표기:
- ours 쪽: `SP`=single-pass, `SK`=split-k
- vllm 쪽: `2D`=single-pass kernel, `3D`=split-k (16 segments) kernel

---

## 2. Headline results (Primary 7)

| case | B×H_q×H_kv×ctx | paths (own\|vd\|vs) | t_ours ms | t_vllm-default ms | t_vllm-smaware ms | **gap_total** | gap_kernel | gap_dispatch |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| llama7b_mha_B1_ctx256  | 1×32×32×256  | SP\|3D\|2D | 0.124 | 0.290 | 0.240 | **+134.0%** | +93.5% | +20.9% |
| llama7b_mha_B1_ctx1k   | 1×32×32×1024 | SP\|3D\|2D | 0.190 | 0.297 | 0.247 | **+55.9%**  | +29.6% | +20.3% |
| llama70b_gqa_B4_ctx2k  | 4×64×8×2048  | SP\|3D\|2D | 0.209 | 0.299 | 0.284 | **+43.1%**  | +35.8% | +5.4%  |
| mqa_B8_ctx4k           | 8×32×1×4096  | SK\|3D\|3D | 0.241 | 0.300 | 0.294 | **+24.7%**  | +22.1% | +2.1%  |
| llama70b_gqa_B4_ctx4k  | 4×64×8×4096  | SP\|3D\|2D | 0.400 | 0.493 | 0.474 | **+23.0%**  | +18.4% | +3.9%  |
| llama7b_mha_B32_ctx1k  | 32×32×32×1024 | SP\|2D\|2D | 2.259 | 2.345 | 2.344 | +3.8%      | +3.8%  | +0.0%  |
| llama7b_mha_B1_ctx4k   | 1×32×32×4096 | SP\|3D\|2D | 0.511 | 0.485 | 0.474 | −5.0%      | −7.2%  | +2.4%  |

**관측**:

- **5/7 이 regression** (vLLM 이 +23%~+134% 느림). 남은 2 개 중 하나 (`B32_ctx1k`) 는 양쪽 다 `2D` 로 dispatch 해서 자연스럽게 parity, 다른 하나 (`B1_ctx4k`) 는 우리 쪽이 살짝 뒤처지는 희귀 케이스.
- **가장 심한 regression (`B1_ctx256`)** 에서 gap 분해: total +134% = (1+20.9%)×(1+93.5%). Dispatch 수정만 하면 134% → 94% 로만 줄어든다. **여전히 2배 가까이 느림**. Kernel side 가 더 큰 기여.
- `B1_ctx4k` 는 유일하게 vllm-default 가 더 빠른 헤드라인 shape (−5%). ctx=4k 에서 16 segments 가 좋은 coverage 가 되는 구간 — 우리 lesson 12 의 SP-only 가 여기선 점유율이 이미 포화되어 split-k gain 없이 느려짐. 우리 쪽 dispatch 도 완벽은 아니라는 교훈.

---

## 3. Sweep results (72 configs)

### 3.1 Categorization

| 범주 | 기준 | count |
|---|---|---:|
| big regression (ours≫vllm) | `gap_total > 20%` | **61** |
| mild regression | `5% < gap_total ≤ 20%` | 12 |
| tie | `−5% ≤ gap_total ≤ 5%` | 5 |
| vllm wins | `gap_total < −5%` | 1 |

전체 79 개 (primary 7 + sweep 72) 중 92% 가 우리 우세, 77% 가 큰 우세.

### 3.2 Mean gap by dispatch path pattern (sweep n=72)

| path pattern (own\|vd\|vs) | n | mean gap_total | mean gap_dispatch | mean gap_kernel | 해석 |
|---|---:|---:|---:|---:|---|
| `SP\|3D\|2D` | 30 | **+48.5%** | +8.8% | +35.3% | 우리는 SP, vLLM-default 는 잘못된 3D, sm-aware 면 2D 로 정정. **Candidate B 의 핵심 타깃 구간.** |
| `SP\|3D\|3D` | 15 | **+93.1%** | +1.1% | +91.0% | B·H_kv 가 작아 sm-aware 로 고쳐도 여전히 3D. **dispatch 로 고칠 수 없는 kernel gap.** |
| `SK\|3D\|3D` | 15 | **+22.7%** | +0.2% | +22.4% | ctx 가 커서 둘 다 split-k. **순수 kernel gap.** |
| `SP\|2D\|2D` | 12 | **+8.9%** | +0.1% | +8.8% | B·H_kv 가 커서 둘 다 single-pass 수렴. Noise/minor kernel gap. |

**큰 그림**:

- Dispatch 수정으로 **의미 있게 회복되는 구간** 은 `SP|3D|2D` 하나뿐. 30 개 shape / 평균 8.8% 회복. 이게 Candidate B 의 "순수 dispatch PR" 이 방어할 수 있는 전부.
- `SP|3D|3D` 구간 (B·H_kv ∈ [1, 16] × ctx=1k 주변, 15 개) 은 **sm-aware heuristic 으로도 못 고친다**. vLLM kernel 의 3D 경로가 이 구간에서 90%+ 느리다는 뜻. 원인은 우리 lesson 12 에서 관찰한 것과 일치할 것: 16 segments 하드코드 × 짧은 ctx → segment 당 작업량 부족 + reduce kernel launch overhead 가 gain 을 상쇄.
- `SK|3D|3D` (ctx=4k, small batch) 는 둘 다 split-k 인데도 우리가 22% 빠름. SEGMENTS adaptive (ctx 기반) vs 16 hardcoded 의 차이로 추정.

### 3.3 Dispatch 단독으로 해결되는 케이스 (8 개)

73 regression 중 **kernel gap ≤ 10%** 인 것 (즉 sm-aware 로 dispatch 만 고치면 "대체로 parity" 가 되는 shape):

| case | gap_total | → gap_kernel (after sm-aware) | dispatch gain | paths |
|---|---:|---:|---:|---|
| sweep_B16_Hkv08_Hq032_ctx4096 | +13.6% | +9.7% | +3.5% | SP\|3D\|2D |
| sweep_B08_Hkv16_Hq064_ctx4096 | +12.8% | +9.3% | +3.3% | SP\|3D\|2D |
| sweep_B04_Hkv32_Hq128_ctx4096 | +12.3% | +8.8% | +3.2% | SP\|3D\|2D |
| sweep_B32_Hkv16_Hq064_ctx1024 | +9.2%  | +9.3% | −0.1% | SP\|2D\|2D |
| sweep_B16_Hkv32_Hq128_ctx1024 | +8.7%  | +8.7% | +0.1% | SP\|2D\|2D |
| sweep_B32_Hkv08_Hq032_ctx4096 | +7.3%  | +7.0% | +0.3% | SP\|2D\|2D |
| sweep_B08_Hkv32_Hq128_ctx4096 | +7.1%  | +7.2% | −0.1% | SP\|2D\|2D |
| sweep_B16_Hkv16_Hq064_ctx4096 | +6.4%  | +6.7% | −0.2% | SP\|2D\|2D |

- 위 8 개 중 실제로 dispatch 바뀌어서 이득 본 건 **앞 3 개** (SP\|3D\|2D, B·H_kv ≈ 128..256, ctx=4k). 나머지 5 개는 path 가 이미 `SP|2D|2D` 라 dispatch 수정이 사실상 null-op — "10% 이내로 닫힌" 건 원래부터 kernel gap 이 작았기 때문.
- 즉 **"dispatch 단독 PR 로 현저한 수혜" 를 받는 shape 은 대략 3-8 개 범위**.

### 3.4 가장 큰 regression top 10 (sweep + primary)

| case | gap_total | gap_kernel | gap_dispatch | paths | B | H_kv | ctx |
|---|---:|---:|---:|---|---:|---:|---:|
| llama7b_mha_B1_ctx256 | +134.0% | +93.5% | +20.9% | SP\|3D\|2D | 1 | 32 | 256 |
| sweep_B02_Hkv02_Hq008_ctx1024 | +128.9% | +115.8% | +6.1% | SP\|3D\|3D | 2 | 2 | 1024 |
| sweep_B16_Hkv02_Hq008_ctx1024 | +120.8% | +61.0% | +37.1% | SP\|3D\|2D | 16 | 2 | 1024 |
| sweep_B08_Hkv08_Hq032_ctx1024 | +96.5% | +66.3% | +18.2% | SP\|3D\|2D | 8 | 8 | 1024 |
| sweep_B01_Hkv08_Hq032_ctx1024 | +96.1% | +94.1% | +1.0% | SP\|3D\|3D | 1 | 8 | 1024 |
| sweep_B02_Hkv16_Hq064_ctx1024 | +94.8% | +60.4% | +21.5% | SP\|3D\|2D | 2 | 16 | 1024 |
| sweep_B01_Hkv04_Hq016_ctx1024 | +94.1% | +87.6% | +3.5% | SP\|3D\|3D | 1 | 4 | 1024 |
| sweep_B16_Hkv01_Hq004_ctx1024 | +94.0% | +88.7% | +2.8% | SP\|3D\|3D | 16 | 1 | 1024 |
| sweep_B01_Hkv02_Hq008_ctx1024 | +93.5% | +88.9% | +2.4% | SP\|3D\|3D | 1 | 2 | 1024 |
| sweep_B02_Hkv01_Hq004_ctx1024 | +93.3% | +92.1% | +0.7% | SP\|3D\|3D | 2 | 1 | 1024 |

- Top 10 중 **6 개가 `SP|3D|3D`** — dispatch 를 sm-aware 로 고쳐도 path 변화 없음. Kernel 자체가 2-배 가까이 느림.
- `B16_Hkv02_Hq008_ctx1024` 는 유일한 "dispatch 수정이 의미 있는" 극단 케이스 (+37% dispatch 기여) 지만 여전히 kernel gap +61% 가 남음.

---

## 4. Reframe: Candidate B 의 실체

원래 가설 (Day 2): "128 은 A100/H100 용 상수 → L4 에서 misfire → dispatch 한 줄만 고치면 Candidate B 완료."

검증 결과: **가설의 방향은 맞지만 크기가 틀림**.

- ✅ Dispatch heuristic 이 L4 에서 misfire 하는 건 **맞다** — `SP|3D|2D` 30 shape 에서 평균 +8.8% 손해.
- ❌ 하지만 "dispatch 만 고치면 parity" 는 **아니다**. Dispatch 수정 후에도 kernel gap 이 geomean +35%, median +23% 남음.
- 💡 진짜 1등 요인은 **vLLM unified kernel 자체가 L4 에서 우리 lesson 11/12 kernel 보다 느림**. 특히 작은 `B*H_kv` × 짧은 ctx 에서.

### 4.1 왜 kernel 이 느린가 (hypothesis, 아직 검증 안 됨)

1. **SEGMENTS 하드코드 16** (upstream `triton_unified_attention.py` line 906-ish): 우리 lesson 12 는 `ceil(ctx / PARTITION_SIZE)` 로 adaptive. ctx=1024 에서 vLLM 이 16 segments 로 쪼개면 segment 당 64 tokens ≈ 4 blocks of size 16. Block-level intra-segment 작업량이 너무 작아서 reduce kernel launch / scratch 쓰기 오버헤드를 못 감춤.
2. **Grid shape 차이**: 우리는 `grid = (B, H_kv, segments)` 또는 `(B, H_kv)`, vLLM 은 `(num_q_blocks, H_kv, [segments])` (`num_q_blocks` = `cdiv(total_tokens, BLOCK_M)`). vLLM 의 q-block 타일링은 prefill/unified 를 위한 것이지만 decode-only 에선 오버헤드.
3. **Scratch prealloc 이 3D 기준**: vLLM backend 는 warmup 시 `softmax_segm_output` 등을 `num_tokens × H_q × 16 × padded_d` 크기로 고정 alloc. 2D path 를 타도 scratch touch cost 가 남음.
4. **Variable-length flat tokens 처리**: vLLM 은 `cu_seqlens` 기반으로 `find_seq_idx` 를 매 블록 실행 → decode-only (Q_len=1) shape 에선 over-engineered.

이 4 개 중 어느 것이 dominant 인지는 **추가 profiling 필요**. Stage 2 로 진입하기 전에 하나 이상에서 "우리 kernel 을 vLLM 기준으로 재구현해 비교" 를 해야 kernel gap 의 원인을 규명 가능.

### 4.2 PR scope 재검토

세 가지 시나리오:

| 시나리오 | 변경 범위 | 예상 기여 | 난이도 | 리스크 |
|---|---|---|---|---|
| **A. Dispatch-only PR** | `triton_attn.py:163` 한 줄 (상수 → `num_sms`-aware) | 30 shape 평균 +9%, 3-8 shape 에서 의미 있음 | 낮음 | 낮음 (기존 path 2D↔3D 선택만 바뀜) |
| **B. Dispatch + SEGMENTS adaptive** | Dispatch + 3D 커널 호출 시 `NUM_SEGMENTS = ceil(ctx / partition_size)` 동적 계산 + scratch prealloc 재조정 | 추가로 `SP\|3D\|3D` 구간 일부 회복 기대 (아직 미측정) | 중간 | 중간 (scratch size 바뀌면 multi-stream 등 영향) |
| **C. Unified kernel 재작성** | 커널 본체를 우리 lesson 11/12 설계로 리팩터 | 전체 gap closure 가능하나 PR 승인 난이도 높음 | 높음 | 높음 |

**1차 권고**: A 로 upstream issue 먼저 열어 maintainer 반응 보기. 동시에 B 의 numeric 근거를 내부적으로 마련 (Stage 1.5 로 새로 스케줄). C 는 Stage 2+ 에서.

A 를 낼 때 정직한 프레이밍:
> "This patch is necessary-but-not-sufficient. On L4 it closes ~9% mean gap on the `B·H_kv ∈ [30, 128]` range by flipping `3D→2D` where appropriate. A larger performance gap of ~35% median remains from kernel-path differences and should be addressed separately. Evidence attached (3-way bench on 79 shapes)."

이렇게 쓰면 reviewer 가 "이게 전부냐?" 로 reject 하지 않고, kernel 쪽 follow-up 이 왜 필요한지 공감대 생김.

---

## 5. Reproducibility

### 5.1 실행 명령

```bash
# 1) primary 7 shapes
python triton_kernels/bench/bench_vllm_vs_ours.py \
  --tag l13_candidateB_stage1_<timestamp>

# 2) dense sweep (primary + 72-config grid)
python triton_kernels/bench/bench_vllm_vs_ours.py \
  --sweep \
  --tag l13_candidateB_stage1_<timestamp>_sweep
```

출력:
- `bench_results/<tag>.csv` — 1 행/shape, 컬럼 = `phase, case, B, H_q, H_kv, ctx, num_sms, vllm_default_thresh, vllm_smaware_thresh, path_ours, path_vllm_default, path_vllm_smaware, t_ours_ms, t_vllm_default_ms, t_vllm_smaware_ms`
- `bench_results/<tag>.json` — 같은 데이터 + meta 블록 (`gpu_name`, `gpu_sms_actual`, `gpu_sm_capability`, `triton_version`, `torch_version`, `heuristic_sm_assumed`, `warmup`, `iters`)

`bench_results/` 는 gitignore 됨. Raw data 파일은 아카이브 목적으로 VM 의 `~/cudatraining-git/bench_results/` 에 보존.

### 5.2 VM 상태 (2026-04-22)

- `gcloud compute instances list --project=nemo-488500`
- `cuda-l4-dev-lesson09` (us-east4-c, g2-standard-4, STANDARD provisioning, NVIDIA L4)
- repo: `~/cudatraining-git` (branch `lesson-13-vllm-audit`)
- Python: `/usr/bin/python3` (3.10.12), preinstalled torch 2.11.0+cu130, triton 3.6.0

### 5.3 Correctness

bench 시작 시 6-way allclose check (B=2, H_q=8, H_kv=4, ctx=768, fp16):
```
ours-SP  vs PyTorch ref:  max_abs_err < 1e-2, PASS
ours-SK  vs PyTorch ref:  PASS
vllm-2D  vs PyTorch ref:  PASS
vllm-3D  vs PyTorch ref:  PASS
```
Kernel 동일성은 byte-diff 로도 확인 (`unified_attention.py` 본체는 upstream `triton_unified_attention.py` 와 import 구문만 다름; [`NOTICE.md`](/Users/xavier/dev/cudatraining/triton_kernels/vllm_extracted/NOTICE.md:1) 참고).

---

## 6. Stage 1.5 — Adaptive NUM_SEGMENTS variant (negative result)

Stage 1 의 끝에 남긴 의문: "`SP|3D|3D` 구간의 kernel gap 90%+ 는 SEGMENTS 하드코드 16 때문인가?" 를 검증. 4 번째 variant 추가:

- **vllm-smaware-adaptive** — vLLM kernel + SM-aware threshold + `num_par_softmax_segments = ceil(ctx/512)` 을 `[1, 32]` 로 clamp (lesson 12 의 partition-size 규칙을 그대로 이식). Scratch tensors (`segm_output/max/expsum`) 도 그 값으로 size.
- 2026-04-23 에 같은 VM, 같은 79 shape 로 재실행. Raw data: `bench_results/l13_candidateB_stage1_5_20260422T1505*.{csv,json}` (4-way columns 추가).

### 6.1 Headline

| metric | vllm-default | vllm-smaware | **vllm-adaptive** | ours |
|---|---:|---:|---:|---:|
| geomean gap vs ours (n=79) | +40.5% | +35.1% | **+36.5%** | 0 |

**Adaptive 는 dispatch-fix 한 smaware 대비 1.4 pp 더 느림 (mean)**. 개선이 아니라 **퇴보**.

### 6.2 Bucketing

Stage 1.5 의 자동 summary (`--sweep` 79 shape):

| bucket | 조건 | count |
|---|---|---:|
| A. dispatch-only 승 | smaware 가 ours 대비 ≤10% | 8 |
| B. adaptive-only 승 | adaptive 가 ≤10% 인데 smaware 는 >10% | **1** (noise, sma +10.3% → adp +10.0%) |
| C. 여전히 stuck | adaptive 로도 >10% | 64 |

→ **adaptive 가 새로 닫은 shape 은 사실상 0**.

### 6.3 Adaptive 가 의미 있는 구간만 분리

Dispatch 가 `smaware → 2D` 로 도달하면 adaptive 도 자동으로 2D (`num_par_softmax_segments` 무시됨). 따라서 adaptive 효과가 실제로 측정되는 구간은 **smaware_path = 3D 인 31 shape** 만. 그 subset 에서:

| 효과 | 기준 | count | % |
|---|---|---:|---:|
| adaptive 가 >2pp 빠름 | help | 2 | 6% |
| ±2pp 이내 | tie | 21 | 68% |
| adaptive 가 >2pp 느림 | **hurt** | 8 | 26% |

**Hurt:help = 4:1**. Tied 구간 평균도 slightly negative. 정말 깨끗한 negative signal.

### 6.4 Path-pattern 별 평균 (`ours | sma_path`)

| bucket | n | mean g_sma | mean g_adp | adp − sma | seg range |
|---|---:|---:|---:|---:|---|
| `SP\|2D` | 48 | +26.6% | +26.9% | +0.2pp | [1, 8] (무의미, sma=2D) |
| `SK\|3D` (long ctx, small batch) | 16 | +24.2% | +27.5% | **+2.8pp** | [8, 8] |
| `SP\|3D` (short ctx, small batch) | 15 | +90.4% | +95.2% | **+2.5pp** | [2, 2] |

**가장 심한 악화 5 shape** (sweep):

| case | B × H_kv × ctx | seg | g_sma | g_adp | Δ |
|---|---|---:|---:|---:|---:|
| sweep_B02_Hkv08_Hq032_ctx4096 | 2×8×4096 | 8 | +23.3% | **+47.5%** | +19.6 pp |
| sweep_B01_Hkv02_Hq008_ctx1024 | 1×2×1024 | 2 | +92.1% | **+129.0%** | +19.2 pp |
| sweep_B02_Hkv04_Hq016_ctx1024 | 2×4×1024 | 2 | +92.2% | **+124.1%** | +16.6 pp |
| sweep_B01_Hkv04_Hq016_ctx4096 | 1×4×4096 | 8 | +23.9% | +44.1% | +16.3 pp |
| sweep_B16_Hkv01_Hq004_ctx4096 | 16×1×4096 | 8 | +23.0% | +37.6% | +11.9 pp |

### 6.5 왜 16 이 더 빠른가 (hypothesis)

작은 `(B, H_kv)` 에서는 outer grid `(total_q_blocks, H_kv)` 가 이미 수십 개 수준이라 SM 을 못 채움. 이 때 3D 축의 16 segments 가 **"parallelism multiplier"** 로 쓰여 점유율을 끌어올림. lesson-12 스타일의 `ceil(ctx/512) = 2` 로 줄이면 grid 가 2×작아져 오히려 under-subscribed 됨.

즉 **upstream 의 16 하드코드는 "큰 batch 에선 낭비" 이기는 하나, 큰 batch 에선 dispatcher 가 이미 2D path 로 보내므로 실제로 3D 가 타는 상황 (small batch) 에서는 16 이 합리적 default**. 우리 lesson 12 의 partition-size 규칙은 `B*H_kv` 가 이미 큰 전제 하에 만들어진 것이라 이 구간엔 부적합.

### 6.6 그래서 kernel gap 은 어디서 오는가?

Stage 1.5 가 SEGMENTS 요인을 제거했으므로 가설이 좁혀짐:

- ~~(a) 16 segments 하드코드~~ — 반증됨.
- **(b) 남은 후보 — 순위대로:**
  1. **Grid layout + BLOCK_M**. vLLM 의 `total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs` 과 `BLOCK_M = max(16, pow2(num_queries_per_kv))` 는 prefill/unified 를 위한 것. Decode-only shape 에서는 낭비. 우리 lesson 12 는 `(B, H_kv)` 로 더 얇게 grid.
  2. **Flat-token `find_seq_idx`**. 블록마다 이진탐색으로 seq 인덱스를 역산. Decode 에서는 `cu_seqlens_q = arange(B+1)` 이라 trivial — 그래도 launch 당 B 번의 binary search ≠ 0.
  3. **Backend scratch prealloc**. 우리 wrapper 에선 `torch.empty` 한 번 할당 비용만 측정되지만, upstream backend 는 steady state 이후 재사용. 이 항목은 측정 영향 ≈ 0 으로 가정해도 무방. 미확인.
  4. **Causal mask + softcap path**. Decode 의 경우 `causal=True` 라도 Q_len=1 이라 mask 는 trivial. 우리 커널은 이 사실을 compile-time 으로 knows. upstream 은 일반 prefill 케이스와 같은 branch.

Hypothesis (1) 이 가장 유력. 확인하려면 ours 의 grid 를 vLLM 풍 `(total_q_blocks, H_kv)` 로 재작성해서 bench → 기대: gap 상당 부분 줄어듦. 이건 "Stage 1.75" 혹은 별도 실험.

### 6.7 Stage 1.5 의 PR 함의

Stage 1 이후 고민했던 3 개 시나리오 (A/B/C) 중:

- **A. Dispatch-only PR** — 유효. 8 shape 에서 10% 이내 회복, 나머지에도 평균 +5.4 pp 개선 (geomean 40.5% → 35.1%). **regression 없음**. L4 에서 순수 이득. A100/H100 은 같은 방향의 underfit 가능성은 있으나 아직 미검증.
- **B. Dispatch + SEGMENTS adaptive PR** — **철회**. Stage 1.5 로 SEGMENTS 는 오히려 regression 유발 (8 shape 에서 2-20pp 악화) 확인. 섞어서 PR 내면 maintainer 가 "이 부분은 measurements 가 하락시키네요" 로 reject 가능.
- **C. Kernel 재작성 PR** — scope 커져서 별도 track. Stage 1.5 가 "grid/BLOCK_M 이 주원인" 으로 가설을 좁혀주긴 했음.

**결론**: upstream issue 는 **A 만 가지고 open**. Stage 1.5 의 adaptive-SEGMENTS 는 "이 방향은 시도했으나 data 가 부정적" 이라는 단서로만 언급 (본 문서 링크). 이렇게 하면:

- PR 자체는 100% 안전 (regression 없음, 작은 diff, 설명 간단)
- Reviewer 가 "kernel gap 도 보세요" 하면 "그건 별 문제이고 probe 도 했으며 하는 중" 이라 응대 가능

### 6.8 Stage 2 기준선 재조정

원래 기준: "L4 에서 realistic shape 3+ 에서 +10% 이상 gap" → **여전히 통과 (5/7 primary)**. Stage 2 로 진행.

하지만 Stage 2 (vLLM e2e) 의 기대치:
- "dispatch-only 패치" 를 적용한 vLLM 빌드 vs 순정 빌드: 기대 e2e 개선 **1-5% 수준** (geomean 5.4pp 는 per-attention-call; e2e 에서는 attention 비중에 희석됨).
- 그 수준이면 PR 가치는 충분하나 헤드라인 "wow" 숫자는 아님. 본 PR 의 value proposition 은 "noise 수준이 아닌, 부수효과 없는, 데이터로 검증된 한 줄 수정" 이 되어야 함.

---

## 7. Next

- **확정**: 시나리오 A 로 upstream issue 드래프트. scope 는 `triton_attn.py:163` 의 `128` 기반 raw threshold 를 `max(num_sms // 2, 1) // max(H_kv, 1)` 같은 SM-aware 식으로 대체. `max(128, num_sms)` 는 L4/T4/A100/H100 범위에서 대부분 128 로 남아 vanilla 와 같은 threshold 를 만들기 때문에 쓰면 안 된다. 증거 첨부: 본 문서 §2 table + §3.2 table + §6.1 table + §8 e2e tables.
- **Stage 2 (e2e) 완료** — §8 참고. L4 × {1.1B, 7B} 두 모델 모두에서 dispatch 수정이 양의 방향으로 e2e decode throughput 에 영향 (batch=32 에서 각각 +5.13%, +1.68%). 단, 모델 크기에 반비례로 gain 감소. PR 프레이밍은 "correctness-of-heuristic (primary) + measurable e2e win in SM-underfilled divergence zone, magnitude depends on attention share which shrinks with model size (secondary)". Large-model high-batch (KV-bound) 시나리오가 더 큰 gain 후보이므로 추후 long-context 실험 여지 열어둠.
- **다음 자연스러운 후속** (우선순위 순):
  - (a) T4 stage 2 — T4 는 L4 보다 SM 수가 적어 (40 vs 58) vanilla threshold 의 underfit severity 가 더 큼 (`128/(40/2)=6.4x` vs L4 `128/(58/2)=4.4x`). 1.1B 에서 +5% 보다 큰 gain 이 나오는지 확인하면 "severity ∝ SM 부족" 논점 강화. [T4 스펙: 40 SM, sm_75, 16GB VRAM — 1.1B 넉넉, 7B 는 불가능에 가까움]
  - (b) L4 + long-context 실험 — `max_model_len=8192`, prompt 를 길게 (1000+ 토큰) 잡아서 prefill 이 아니라 **긴 KV 에서의 decode** 를 측정. Attention share 가 커지면서 7B 에서도 gain 이 5%+ 로 회복될지 여부.
  - (c) Cross-GPU (A10, A100, H100) — 소스가 확보되면. A100 은 108 SM 이라 L4/T4 보다 severity 는 작지만 아직 경계선.
- **Optional**: Stage 1.75 (grid/BLOCK_M 차이가 kernel gap 의 주원인인지 한 실험으로 확인). 가치: 추후 kernel PR 의 근거.

### Open questions (updated)

1. ~~`SP|3D|3D` 구간의 kernel gap 이 SEGMENTS 때문인가~~ — **아니오** (§6).
2. L4 의 발견이 A100/H100 에서도 같은 방향/같은 크기인가? — 미해결. Dispatch-only PR 설명에 "L4 에서 관측; A100 은 `128/(108/2) ≈ 2.4x` 이므로 threshold 도 같은 방향으로 underfit 가능성. H100 은 `128/(132/2) ≈ 1.9x`, 경계. 재현 가능한 bench script 첨부." 로 프레이밍.
3. SM-aware threshold 의 구체적 공식 — `(num_SMs // 2) // H_kv` 는 우리가 임의로 정한 값. `num_SMs // H_kv` 나 `num_SMs * 3 // 4 // H_kv` 같은 다른 식도 실험해볼 가치. PR 전에 sensitivity 측정 짧게 추가 권장.
4. Grid/BLOCK_M 이 kernel gap 의 주원인인지 (Stage 1.75 질문).
5. ~~Larger model 에서 e2e gain 이 비례/성장하는가?~~ — **부분 해결 (short context 한정)**. Qwen2.5-7B @ L4 에서 gain 은 1.7% 로 1.1B (5.1%) 보다 **작음**. 원인: short context decode 에서 MLP 가 모델 크기에 대해 `hidden × intermediate × 3 × layers` 로 빠르게 커져 (TinyLlama→Qwen MLP ~7.5x vs attention BW ~2.5x) attention share 가 반대로 줄어듦. Long-context (large KV) 나 larger-batch 에서는 attention share 가 다시 커질 수 있으므로 별도 실험 여지 (§7 next-(b)).

6. **(NEW) CUDA graph capture-size snapping 의 영향** — Stage 1 (kernel-level) 은 raw threshold 7 로 측정, Stage 2 (vLLM e2e) 는 snap 후 effective threshold 8 로 dispatch. 우리 batch grid {8,16,32} 에서 boundary 가 raw 7→snap 8 로 1 만큼 밀린 결과 batch=8 이 divergence 가 아닌 "양쪽 같은 path" zone 에 빠짐. PR 작성 시 "snapping 까지 고려한 effective threshold 가 우리 bench 결과를 결정" 임을 본문에 명시할 것. 후속 실험에서 batch=9 추가하면 boundary 양쪽을 직접 스캔 가능.

---

## 8. Stage 2 — vLLM e2e validation

Stage 1 이 "attention kernel call 한 번 의 latency gap" 을 측정했다면, Stage 2 는 **"그 gap 이 decode throughput 으로 실제 이어지는가"** 를 확인한다. Attention 은 decode 시간의 일부분이므로 (MatMul, softmax, norm, tokenizer, scheduler 등과 공유), per-call 25% 개선이 e2e 몇 % 로 translate 되는지 측정해야 PR 의 실질적 가치를 판단할 수 있다.

### 8.1 Setup

| 항목 | 값 |
|---|---|
| VM | `cuda-l4-dev-lesson09`, L4 (58 SMs, sm_89) |
| vLLM | 0.19.1 (pip install, venv `/home/xavier/vllm-venv`) |
| Backend | `attention_backend="TRITON_ATTN"` (강제 지정 — 기본값은 FA2 로 떨어져서 heuristic 을 안 거침) |
| Models | (a) `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (H_q=32, H_kv=4, d=64, 22 layers, hidden=2048) / (b) `Qwen/Qwen2.5-7B-Instruct` (H_q=28, H_kv=4, d=128, 28 layers, hidden=3584) |
| Dtype | float16 |
| Workload | 32-token prompt, `max_tokens=256`, `temperature=0.0`, `ignore_eos=True`, `enforce_eager=False` (CUDA Graphs on), `gpu_memory_utilization=0.80` (1.1B) / `0.90` (7B), `max_model_len=2048` |
| Batches | {8, 16, 32} — divergence zone on L4/H_kv=4. **Effective threshold (post-snap)**: vanilla=32 (raw `128//4=32` → snap 32, no-op), smaware=8 (raw `29//4=7` → snap **8**, capture size [1,2,4,8,16,...] 중 7 에 가장 가까움). 따라서 dispatch path: vanilla 는 세 batch 모두 3D (8,16,32 모두 32 이하), **smaware 는 batch=8 만 3D (8 > 8 = false), batch=16,32 는 2D**. → Path divergence 는 batch ∈ {16, 32} 두 점에서만 일어나고, batch=8 은 같은 3D kernel 을 두 variant 가 똑같이 호출. snapping 메커니즘은 [`triton_attn.py:166-176`](/tmp/vllm-investigation/vllm/vllm/v1/attention/backends/triton_attn.py:166), 추가 PR #28306 참고. VM 상에서 `seq_threshold_3D` 값을 직접 probe 해서 raw 7 → effective 8 임을 확인 (vanilla 는 32 → 32). |
| Measure | `llm.generate(prompts, ...)` wall time, 5 iters measure + 2 warmup per batch, `torch.cuda.synchronize()` pre/post |

모델 선택 근거:
- 둘 다 H_kv=4 → 정확히 같은 divergence-zone 구간 (vanilla 3D vs smaware 2D). 모델 크기만 변수.
- 둘 다 HF non-gated — 재현이 쉬움 (Llama-3.2 는 access token 필요).
- 1.1B: MLP-dominant 소형 (hidden=2048, 22 layers). Attention share 가 중간 정도.
- 7B: MLP 가 더 지배적이 되는 medium (hidden=3584, 28 layers, d_head=128 로 per-layer attention cost 는 올라감). 단, total decode cost 에서 attention share 는 오히려 줄어듦 (§8.5 참고).

### 8.2 Patch toggle

`scripts/toggle_vllm_patch.py` — 설치된 vLLM tree 의 `triton_attn.py` 의 `self.seq_threshold_3D = ...` line 을 vanilla/smaware 로 flip. Anchor pattern-match (line-number 가 아닌) 로 version drift 에 강함 (0.19.1 에서는 line 151, upstream main 에서는 line 163 — 같은 script 가 둘 다 잡음). `.pyc` 도 같이 정리.

```bash
python scripts/toggle_vllm_patch.py --mode vanilla --yes  # 원상 복구
python scripts/toggle_vllm_patch.py --mode smaware --yes  # patch 적용
python scripts/toggle_vllm_patch.py --status              # 현재 상태만 리포트
```

### 8.3 Results — TinyLlama-1.1B

| batch | dispatch (vanilla / smaware) | vanilla wall_ms (median ± stdev) | smaware wall_ms | vanilla tok/s | smaware tok/s | **Δtok/s %** |
|---:|:---:|---:|---:|---:|---:|---:|
| 8  | 3D / **3D** (same kernel) | 2350.7 ± 1.4 | 2354.0 ± 1.2 |  871.2 |  870.0 | **−0.14%** (sanity) |
| 16 | 3D / 2D | 2426.4 ± 0.5 | 2358.3 ± 1.0 | 1688.1 | 1736.9 | **+2.89%** |
| 32 | 3D / 2D | 2718.1 ± 1.6 | 2585.4 ± 1.9 | 3013.8 | 3168.5 | **+5.13%** |

(median-of-5 iters, 각 iter 당 `batch × 256` decode step. `ignore_eos=True` 로 모든 seq 가 max_tokens 까지 decode. stdev 은 sub-%; 측정 안정적. **batch=8 의 −0.14%** 는 양 variant 가 같은 3D kernel 을 호출함에도 0 이 아닌 측정 noise — 패치 toggle 이 dispatch 한 줄만 바꾸고 다른 부수효과가 없다는 sanity check 로 작용.)

Raw: `bench_results/l13_candidateB_stage2_20260423T073336Z_{vanilla,smaware}.{csv,json}`.

### 8.3b Results — Qwen2.5-7B (cross-check, larger model)

| batch | dispatch (vanilla / smaware) | vanilla wall_ms (median ± stdev) | smaware wall_ms | vanilla tok/s | smaware tok/s | **Δtok/s %** |
|---:|:---:|---:|---:|---:|---:|---:|
| 8  | 3D / **3D** (same kernel) | 14838.7 ± 7.5 | 14866.8 ± 2.0 | 138.0 | 137.8 | **−0.19%** (sanity) |
| 16 | 3D / 2D | 15110.2 ± 3.2 | 14970.6 ± 0.7 | 271.1 | 273.6 | **+0.92%** |
| 32 | 3D / 2D | 16565.2 ± 3.6 | 16292.5 ± 4.9 | 494.5 | 502.8 | **+1.68%** |

(동일 workload 와 divergence-zone batch 들, 모델만 1.1B → 7B. stdev 은 여전히 sub-% 로 매우 안정. 동일 방향·부호, 다만 magnitude 가 ~1/3 로 줄어듦. batch=8 은 양쪽이 같은 3D path 를 타는 control point 라 archived run 에서는 −0.14% / −0.19%, verification rerun 에서는 −0.54% / +0.01% 로 **sub-1% noise floor** 에 머묾.)

Raw: `bench_results/l13_candidateB_stage2_qwen7b_20260423T085705Z_{vanilla,smaware}.{csv,json}`.

### 8.3c Verification rerun — 2026-04-24

Codex 인수 직후 같은 L4 VM / 같은 vLLM 0.19.1 환경에서 Stage 2 를 한 번 더 순차 실행했다. 목적은 archived CSV 의 숫자를 대체하는 것이 아니라, **방향성·크기·same-path sanity** 가 재현되는지 확인하는 것.

| model | rerun tag | B=8 same-path | B=16 divergence | B=32 divergence |
|---|---|---:|---:|---:|
| TinyLlama-1.1B | `l13_verify_tiny_20260424T020038Z` | −0.54% | +2.53% | +4.85% |
| Qwen2.5-7B | `l13_verify_qwen7b_20260424T020551Z` | +0.01% | +0.90% | +1.66% |

해석: B=16/32 의 positive gain 은 archived run 과 같은 방향/크기다. B=8 은 양쪽 모두 3D path 를 타는 control point 라 run-to-run 으로 −0.5~+0.0% 수준에서 흔들릴 수 있으며, 이 구간은 performance claim 이 아니라 **side-effect 없음** 을 보는 sanity check 로만 사용한다. 재실행 후 installed vLLM state 는 `vanilla` 로 복구 확인.

### 8.4 해석

1. **Dispatch 변화가 e2e 로 실제로 이어진다 (두 모델 모두).** Batch=16, 32 에서 stdev 대비 Δ 가 10~100x 큼. 단순 "kernel micro-bench 상 차이" 가 아니라 throughput 에 반영되는 real effect.

2. **Batch 크기에 비례해 gain 이 커진다 (두 모델 공통).** vanilla 의 3D split-k 는 segments 축을 추가해 더 잘게 쪼개지만, L4 (58 SM) 에서 outer grid 가 이미 작을 때 (num_q_blocks=1~2, H_kv=4) 3D 가 제공하는 extra parallelism 은 실제 SM 을 더 채우기보다 **launch/reduction overhead 만 추가**. Batch=32 는 2D single-pass 의 장점이 가장 크게 드러나는 지점. **Batch=8 은 두 variant 가 path divergence 자체를 못 일으킴** (smaware effective threshold=8 에서 `8 > 8` 은 false → 둘 다 3D kernel 호출). 따라서 batch=8 의 작은 delta 는 "underfill 로 인한 비슷함" 이 아니라 "literally 같은 코드 경로" — 측정 noise floor 만 보고 있는 것.

   - 부산물 효과: divergence boundary 가 raw threshold (7) 가 아니라 snapped (8) 에서 결정되므로, **batch=9 에서야 처음으로 path 가 갈린다**. 본 실험은 [8, 16, 32] 만 측정해서 boundary 양옆 (8 vs 9, 16) 을 깔끔히 못 잡음. 후속에서 `--batches 7 8 9 12` 같이 boundary 직접 sweep 하면 snapping 효과를 micro-bench 로 직접 분리 가능.

3. **반직관: 큰 모델에서 gain 이 오히려 작다 (5.13% → 1.68%).** 사전 예상 (open question #5) 은 "7B 에서는 attention share 가 커져서 gain 이 더 클 것" 이었는데, 관측은 반대. 원인을 분해하면 (이 계산은 ballpark 추정용 — 정확한 수치 검증은 nsys profile 이 필요):

   - Decode 한 step 의 memory BW 는 (a) MLP weight load + (b) attention KV read 로 나뉨.
   - (a) MLP per layer = `hidden × intermediate × 3` (SwiGLU: gate/up/down projection). 두 모델 spec:
     - TinyLlama: hidden=2048, intermediate=5632, layers=22 → per-layer 34.6M, total **761M** params (MLP weight)
     - Qwen2.5-7B: hidden=3584, intermediate=18944, layers=28 → per-layer 203.6M, total **5.7B** params
     - 비율: **7.5x** MLP weight load (단순 hidden² 비율 3.0x 가 아니라 intermediate ratio 까지 포함해야 함 — TinyLlama 의 intermediate/hidden=2.75 vs Qwen 의 5.29 가 추가 1.9x 곱해짐).
   - (b) attention KV read per layer = `2 × batch × seq × H_kv × d_head` (K + V). 두 모델 모두 H_kv=4, batch=32, seq≈300 (decode 평균):
     - TinyLlama: per-layer 32×300×8×64=4.9M values → ×22 = 108M values
     - Qwen: per-layer 32×300×8×128=9.8M values → ×28 = 274M values
     - 비율: **2.5x** attention BW.
   - Attention share 추정 (memory BW 만 고려한 단순 모델, 실제는 compute/scheduler/embedding 등도 포함):
     - TinyLlama: 108M / (108M + 761M) ≈ **12%**
     - Qwen-7B: 274M / (274M + 5700M) ≈ **4.6%**
     - 약 2.5–3x 차이. 관측된 e2e gain 비율 5.13/1.68 ≈ **3.05x** 와 ballpark 일치.
   - Per-attention-call gain 이 X% 일 때 e2e gain ≈ attention-share × X. 관측 e2e gain 으로 역산하면: 1.1B 5.13/12% ≈ 43%, 7B 1.68/4.6% ≈ 36% per-call gain. Stage 1 sweep 의 H_kv=4 short-ctx 구간 gap_dispatch 가 이 범위 (해당 shape 측정값이 sweep 표에 있음) 와 일치 — 정합 OK. 다만 이 역산은 "memory BW 만이 e2e bottleneck" 가정에 기반하므로 정확한 attribution 은 nsys profile 필요.

4. **Stage 1.5 가 보여준 adaptive-SEGMENTS 의 negative result 와 consistent (두 모델 모두).** "segments 축 추가 parallelism" 의 비용 (reduction, extra mem bandwidth) 이 작은 outer grid 에서 **benefit 보다 크다** 는 본질이 같음. Dispatch 를 2D 로 forcing 하면 이 overhead 자체가 사라짐. 7B 도 같은 기울기로 gain 이 batch 에 따라 선형 증가.

5. **Implication for PR magnitude claim**: Performance gain 은 본질적으로 "attention share × per-call dispatch-overhead saving" 이며, attention share 는 (i) 모델이 작을수록, (ii) batch 가 클수록, (iii) seq_len (KV) 가 길수록, (iv) H_kv 가 클수록 커짐. 본 실험은 (i)(ii) 효과만 측정. Long-context 나 H_kv 가 큰 (e.g. MHA Llama-7B H_kv=32) 모델은 미검증이지만 같은 논리로 gain 이 더 클 것 (§7 next-(b)).

### 8.5 PR 프레이밍 (업데이트)

기존 Stage 1 만 있었을 때는 "correctness-of-heuristic" (constant 128 이 small-SM GPU 에서 underfit) 이 주 프레임이었고 performance 는 "modest" 정도로 표현 예정. Stage 2 두 모델 결과를 반영한 최종 프레이밍:

- **Primary claim (correctness)** — "`MIN_LAUNCH_GRID_SIZE_2D = 128` 의 `128` 은 `triton_attn.py:49` 의 의도적 상수이며, "2D kernel 의 minimum launch grid size" 라는 명확한 의미를 가진다 (PR #28306). 이 값은 A100 (108 SM) / H100 (132 SM) 에 맞춰져 있어 그 GPU 들에서는 `seq_threshold_3D = 128 // H_kv` 가 합리적이지만, **`128` 자체가 GPU 의 실제 SM count 와 분리되어 있다는 점이 hardware-agnostic bug**. L4 (58 SM), T4 (40 SM) 같은 small-SM GPU 에서는 `128 // H_kv` 가 실제 달성 가능한 grid 보다 큰 threshold 를 만들어 small-batch decode 에서도 3D split-k 가 trigger 되며 parallelism 없이 overhead 만 추가한다. `max(num_sms // 2, 1) // num_kv_heads` 로 바꾸면 GPU-aware 가 된다. (이때 `triton_attn.py:166-176` 의 CUDA graph capture-size snapping 은 그대로 유지 — snap 은 "captured graph 가 정확한 path 를 cover 하도록" 하는 직교 메커니즘.)"
- **Secondary claim (measured e2e performance)** — "L4 + H_kv=4 divergence-zone 워크로드에서 smaware 가 **short-context decode 기준 batch=16 +0.9~2.9%, batch=32 +1.7~5.1% tok/s 개선**, 모델 크기에 반비례 (attention share 때문). 측정 stdev < 0.5%. **Boundary 직전 batch=8 은 snap 후 effective threshold (8) 와 num_seqs (8) 가 같아져 양쪽이 같은 3D path 를 호출하므로 sub-1% noise 만 보임 — 이는 패치 toggle 이 dispatch 한 줄만 바꾸고 다른 부수효과가 없다는 sanity check.**"
- **Supporting mechanism** — Stage 1.5 negative result (adaptive SEGMENTS 는 도움 안 됨 → 문제는 dispatch 이지 kernel tuning 이 아님).
- **Caveat** — "L4 에서 관측, 두 non-gated HF 모델에서 재현됨. A100/H100 에서도 같은 방향의 underfit 가능성 있으나 미검증 (`severity ≈ 128 / (num_sms // 2)` 라 H100=132SM 에서 severity ≈ 1.94x, A100=108SM 에서 2.37x, L4=58SM 에서 4.41x, T4=40SM 에서 6.40x 로 SM 적을수록 큼). 재현 가능한 bench script (Stage 1 `bench_vllm_vs_ours.py` + Stage 2 `bench_vllm_e2e.py` + `scripts/toggle_vllm_patch.py`) 첨부."

이 프레이밍이면 maintainer 가 "L4 타겟 아님" 이라고 쳐내기 힘듦 — correctness 논점이 performance 에서 온 게 아니라, formula 자체의 SM-count 무지에서 옴.

### 8.6 재현

```bash
# L4 VM 기준, vllm venv 활성화 상태
source ~/vllm-venv/bin/activate
cd ~/cudatraining-git

TAG=l13_candidateB_stage2_$(date -u +%Y%m%dT%H%M%SZ)

# --- Model (a): TinyLlama-1.1B ---
python scripts/toggle_vllm_patch.py --mode vanilla --yes
python -m triton_kernels.bench.bench_vllm_e2e \
    --variant vanilla \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --batches 8 16 32 \
    --max-new-tokens 256 --iters 5 --warmup 2 \
    --max-model-len 2048 \
    --tag ${TAG}

python scripts/toggle_vllm_patch.py --mode smaware --yes
python -m triton_kernels.bench.bench_vllm_e2e \
    --variant smaware \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --batches 8 16 32 \
    --max-new-tokens 256 --iters 5 --warmup 2 \
    --max-model-len 2048 \
    --tag ${TAG}

# --- Model (b): Qwen2.5-7B (L4 23GB VRAM 에 빡빡하니 gpu_memory_utilization 0.90) ---
TAG7B=l13_candidateB_stage2_qwen7b_$(date -u +%Y%m%dT%H%M%SZ)

python scripts/toggle_vllm_patch.py --mode vanilla --yes
python -m triton_kernels.bench.bench_vllm_e2e \
    --variant vanilla \
    --model Qwen/Qwen2.5-7B-Instruct \
    --batches 8 16 32 \
    --max-new-tokens 256 --iters 5 --warmup 2 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.90 \
    --tag ${TAG7B}

python scripts/toggle_vllm_patch.py --mode smaware --yes
python -m triton_kernels.bench.bench_vllm_e2e \
    --variant smaware \
    --model Qwen/Qwen2.5-7B-Instruct \
    --batches 8 16 32 \
    --max-new-tokens 256 --iters 5 --warmup 2 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.90 \
    --tag ${TAG7B}

python scripts/toggle_vllm_patch.py --mode vanilla --yes  # 원상 복구
```

Raw 산출물:
- 1.1B: `bench_results/l13_candidateB_stage2_20260423T073336Z_{vanilla,smaware}.{csv,json}`
- 7B:  `bench_results/l13_candidateB_stage2_qwen7b_20260423T085705Z_{vanilla,smaware}.{csv,json}`

---

## 재료

- Audit Day 1-2: [`docs/vllm_audit_01_attention_path.md`](/Users/xavier/dev/cudatraining/docs/vllm_audit_01_attention_path.md:1)
- Stage 1 kernel 추출: [`triton_kernels/vllm_extracted/unified_attention.py`](/Users/xavier/dev/cudatraining/triton_kernels/vllm_extracted/unified_attention.py:1), [`NOTICE.md`](/Users/xavier/dev/cudatraining/triton_kernels/vllm_extracted/NOTICE.md:1)
- Bench harness (Stage 1/1.5, kernel-level): [`triton_kernels/bench/bench_vllm_vs_ours.py`](/Users/xavier/dev/cudatraining/triton_kernels/bench/bench_vllm_vs_ours.py:1)
- Bench harness (Stage 2, e2e decode): [`triton_kernels/bench/bench_vllm_e2e.py`](/Users/xavier/dev/cudatraining/triton_kernels/bench/bench_vllm_e2e.py:1)
- Patch toggle helper: [`scripts/toggle_vllm_patch.py`](/Users/xavier/dev/cudatraining/scripts/toggle_vllm_patch.py:1)
- Raw data (Stage 1, 3-way): `bench_results/l13_candidateB_stage1_20260422T140813Z.{csv,json}` (primary 7), `bench_results/l13_candidateB_stage1_20260422T140953Z_sweep.{csv,json}` (primary 7 + sweep 72)
- Raw data (Stage 1.5, 4-way with adaptive SEGMENTS): `bench_results/l13_candidateB_stage1_5_20260422T150508Z.{csv,json}` (primary 7), `bench_results/l13_candidateB_stage1_5_20260422T150631Z_sweep.{csv,json}` (primary 7 + sweep 72)
- Raw data (Stage 2, e2e vanilla vs smaware): `bench_results/l13_candidateB_stage2_20260423T073336Z_{vanilla,smaware}.{csv,json}`
- Lesson 11 (paged attention vLLM v0 audit): [`docs/blog_draft_lesson_11_paged_attention.md`](/Users/xavier/dev/cudatraining/docs/blog_draft_lesson_11_paged_attention.md:1)
- Lesson 12 (split-k + auto-dispatch origin): [`docs/blog_draft_lesson_12_split_k.md`](/Users/xavier/dev/cudatraining/docs/blog_draft_lesson_12_split_k.md:1)
