# vLLM Audit 02 — Candidate B Stage 1: Kernel-Level Bench (L4, sm_89)

Lesson 13, Week 2 Day 1-2. Candidate B (`seq_threshold_3D = 128 // num_kv_heads` dispatch heuristic) 의 Stage 1 검증 결과.

- 실행: 2026-04-22, GCP L4 VM `cuda-l4-dev-lesson09` (nemo-488500 / us-east4-c, NVIDIA L4, 58 SMs, sm_89, torch 2.11.0+cu130, triton 3.6.0)
- 산출물: [`bench_vllm_vs_ours.py`](/Users/xavier/dev/cudatraining/triton_kernels/bench/bench_vllm_vs_ours.py:1), [`vllm_extracted/unified_attention.py`](/Users/xavier/dev/cudatraining/triton_kernels/vllm_extracted/unified_attention.py:1), raw CSV/JSON at `bench_results/l13_candidateB_stage1_20260422T140*.{csv,json}`

---

## TL;DR

1. **vLLM Triton unified attention 은 L4 에서 우리 lesson 12 kernel 대비 shape 의 대다수에서 느리다.** 79 config 중 73 개 (92%) 에서 우리가 5% 이상 빠르고, 61 개 (77%) 에서 20% 이상 빠르다. Geomean total gap **+40.9%**, median **+23.8%**.
2. **원인 분해 결과, 문제는 "dispatch heuristic 하나" 가 아니다.** Dispatch 를 SM-aware 로 고치면 geomean **+4.1%** / median **+2.4%** 밖에 회복 안 된다. 나머지 (geomean +35.4% / median +23.2%) 는 **kernel 자체** 의 gap 이다.
3. **Dispatch 수정만으로 10% 이내로 닫히는 shape 은 73 개 regression 중 8 개 (11%).** 65 개 (89%) 는 dispatch 를 고쳐도 여전히 10%+ 뒤처진다.
4. 따라서 **Candidate B 는 "깔끔한 heuristic PR" 이 아니라 "heuristic + kernel" 두 축으로 봐야** 한다. PR story 를 다음 섹션에서 재구성.

---

## 1. Methodology

### 1.1 3-way bench design

각 shape 마다 같은 입력으로 세 경로를 돌리고 median-of-50 latency (ms) 를 측정.

| Variant | Kernel | Dispatch threshold | 역할 |
|---|---|---|---|
| **ours** | Lesson 12 single-pass + split-k paged attention (`triton_kernels/paged_attention.py`) | 우리 heuristic: `B*H_kv < 0.5*SM ∧ segments ≥ 4 → SK, else SP` | baseline (우리 구현) |
| **vllm-default** | `triton_kernels/vllm_extracted/unified_attention.py` (upstream byte-identical) | 상수 `seq_threshold_3D = 128 // H_kv` (upstream `triton_attn.py:163`) | "현재 upstream" |
| **vllm-smaware** | 동일 vLLM kernel | `seq_threshold_3D = (num_SMs // 2) // H_kv` (L4→29, A100→54, H100→66) | "dispatch 만 고친 vLLM" |

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

## 6. Next

Stage 1 결과를 받아 판단할 것:

- **Stage 2 (vLLM e2e bench) 로 진행하는가?**
  - 기준선 (Day 2 에서 설정): "L4 에서 realistic shape 3 개 이상에 +10% 이상 latency 차이" → **5/7 primary shape 에서 초과**. 기준 통과.
  - 하지만 원래 plan 은 "dispatch 한 줄" 전제였음. 현재 결과 (kernel gap 이 dominant) 를 보면 Stage 2 의 e2e 기대치를 재설정해야 함 — "dispatch 만 고친 빌드" vs "ours kernel 삽입 빌드" 를 다르게 측정.
- **Stage 1.5 (SEGMENTS adaptive 추가 bench) 를 끼워넣을까?**
  - `vllm-smaware` variant 옆에 `vllm-smaware-adaptive-segments` 를 추가하면 4-way. `SP|3D|3D` 구간에서 어떻게 움직이는지만 봐도 "kernel gap 의 일부가 SEGMENTS 였다" 가 분리됨. 예상 작업량 0.5 일.
  - Stage 1.5 결과가 크면 Candidate B 를 "dispatch + SEGMENTS" 패키지로 승격 후 Stage 2 로. 작으면 순수 dispatch PR 로 가고 kernel 은 별 이슈로.

- **Upstream issue drafting 시점**
  - 권장: Stage 1.5 까지 끝낸 후, "dispatch + SEGMENTS" 의 측정치를 붙여서 issue 를 열기. 순수 dispatch-only issue 로는 maintainer 가 "이게 전부냐?" 반응할 가능성 높음.

### Open questions

1. `SP|3D|3D` 구간의 kernel gap 이 정확히 어디서 오는가 (SEGMENTS vs grid shape vs scratch vs flat-token)? → Stage 1.5 또는 별도 profiling (NCU) 필요.
2. L4 의 발견이 A100/H100 에서도 같은 방향/같은 크기인가? SM count scaling 만 보면 A100 은 `128/108 ≈ 1.19×` 살짝 안 맞고, H100 은 `128/132 ≈ 0.97×` 거의 맞음. 즉 **L4 와 A100 에서는 misdispatch, H100 은 잘 맞음** 이 upstream 설계 의도였을 가능성. A100 bench 없이는 반론 못 막음.
3. `vllm-smaware` 의 threshold 를 `num_SMs` 그대로 (우리 현재 `num_SMs // 2` 가 아니라) 썼을 때는 어떤가? Baseline 이 하나 더 있으면 threshold 함수의 민감도도 알 수 있음.

---

## 재료

- Audit Day 1-2: [`docs/vllm_audit_01_attention_path.md`](/Users/xavier/dev/cudatraining/docs/vllm_audit_01_attention_path.md:1)
- Stage 1 kernel 추출: [`triton_kernels/vllm_extracted/unified_attention.py`](/Users/xavier/dev/cudatraining/triton_kernels/vllm_extracted/unified_attention.py:1), [`NOTICE.md`](/Users/xavier/dev/cudatraining/triton_kernels/vllm_extracted/NOTICE.md:1)
- Bench harness: [`triton_kernels/bench/bench_vllm_vs_ours.py`](/Users/xavier/dev/cudatraining/triton_kernels/bench/bench_vllm_vs_ours.py:1)
- Raw data: `bench_results/l13_candidateB_stage1_20260422T140813Z.{csv,json}` (primary 7), `bench_results/l13_candidateB_stage1_20260422T140953Z_sweep.{csv,json}` (primary 7 + sweep 72)
- Lesson 11 (paged attention vLLM v0 audit): [`docs/blog_draft_lesson_11_paged_attention.md`](/Users/xavier/dev/cudatraining/docs/blog_draft_lesson_11_paged_attention.md:1)
- Lesson 12 (split-k + auto-dispatch origin): [`docs/blog_draft_lesson_12_split_k.md`](/Users/xavier/dev/cudatraining/docs/blog_draft_lesson_12_split_k.md:1)
