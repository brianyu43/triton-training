# Lesson 08 Handoff — Triton 포팅 (Scope B, 4 커널)

기준 날짜: `2026-04-19`

주제: **레슨 1-6 의 CUDA 커널을 Triton 으로 재작성. "추상화의 비용" 을 수치로 측정.**

세션 규모: 6 시간 (Phase 0-6), Scope B (reduction / softmax / matmul / flash attention).

## 1. 이번 세션에서 한 일

### 하드웨어 업그레이드

- 레슨 1-7 은 **T4 (sm_75, Turing)** 에서 돌렸음 — Tensor Core 가 FP16 WMMA 한 종류, TF32 없음.
- 이번엔 **L4 (sm_89, Ada Lovelace)** 로 올림:
  - TF32 TC 121 TFLOPS, FP16 TC 242 TFLOPS, FP8 TC 485 TFLOPS
  - HBM 300 GB/s, L2 cache 48 MB (T4 의 8 배)
  - GCP `g2-standard-4` SPOT, us-west4-a, **`cuda-l4-dev-lesson08`**
- 이유: Triton `tl.dot` 가 TF32 를 자동 쓰는 걸 측정하려면 TF32 TC 가 있는 GPU 필요.

### Triton 4 커널 포팅

한 파일당 ~40-100 줄. 전부 `@triton.autotune + @triton.jit` 구조:

| 커널 | 파일 | 줄수 | 핵심 |
|---|---|---|---|
| reduction | [triton_kernels/reduction.py](/Users/xavier/dev/cudatraining/triton_kernels/reduction.py:1) | ~90 | 2-pass (`tl.sum` partial → torch final), autotune on BS+warps |
| softmax | [triton_kernels/softmax.py](/Users/xavier/dev/cudatraining/triton_kernels/softmax.py:1) | ~75 | 1 program/row, `BLOCK_SIZE = next_pow2(N)`, autotune on warps |
| matmul | [triton_kernels/matmul.py](/Users/xavier/dev/cudatraining/triton_kernels/matmul.py:1) | ~130 | Grouped program ID swizzling, `tl.dot` → TF32/fp16 TC 자동 |
| flash attention | [triton_kernels/flash_attention.py](/Users/xavier/dev/cudatraining/triton_kernels/flash_attention.py:1) | ~100 | Online softmax in K/V loop, 2×`tl.dot` per iter |

### 벤치 매트릭스

각 커널당 3-4 way 비교 (우리 CUDA / torch / Triton) + correctness + autotune config 기록:

| 파일 | 샘플 shape | CSV |
|---|---|---|
| [bench/bench_reduction.py](/Users/xavier/dev/cudatraining/triton_kernels/bench/bench_reduction.py:1) | n = 1M/4M/16M/67M | `reduction_3way.csv` + `reduction_triton_sweep.csv` |
| [bench/bench_softmax.py](/Users/xavier/dev/cudatraining/triton_kernels/bench/bench_softmax.py:1) | (M,N) 4 shapes | `softmax_4way.csv` + `softmax_triton_sweep.csv` |
| [bench/bench_matmul.py](/Users/xavier/dev/cudatraining/triton_kernels/bench/bench_matmul.py:1) | 512/1024/2048/4096³ × {fp32,fp16} | `matmul_3way_fp32.csv`, `matmul_3way_fp16.csv` |
| [bench/bench_flash_attention.py](/Users/xavier/dev/cudatraining/triton_kernels/bench/bench_flash_attention.py:1) | N=512/1k/2k/4k/8k, d=64 × {fp32,fp16} | `flash_attention_4way.csv` |

### GCP 자동화

Phase 별 독립 runner — `scripts/gcp_run_lesson08_phase{0,1,2,3,4}.sh`. 패턴: 레포 복사 → sm_89 재빌드 → 벤치 → CSV 회수.

### 블로그 초안

[docs/blog_draft_triton_vs_cuda.md](/Users/xavier/dev/cudatraining/docs/blog_draft_triton_vs_cuda.md:1) — 8 섹션, ~3500 단어. 제목: *"Triton vs CUDA: 추상화의 비용은 어디서 나타나는가"*.

## 2. 산출물

Triton 커널 (5 파일, smoke 포함):
- [triton_kernels/smoke_vector_add.py](/Users/xavier/dev/cudatraining/triton_kernels/smoke_vector_add.py:1)
- [triton_kernels/reduction.py](/Users/xavier/dev/cudatraining/triton_kernels/reduction.py:1)
- [triton_kernels/softmax.py](/Users/xavier/dev/cudatraining/triton_kernels/softmax.py:1)
- [triton_kernels/matmul.py](/Users/xavier/dev/cudatraining/triton_kernels/matmul.py:1)
- [triton_kernels/flash_attention.py](/Users/xavier/dev/cudatraining/triton_kernels/flash_attention.py:1)

벤치 (4 파일):
- [triton_kernels/bench/bench_reduction.py](/Users/xavier/dev/cudatraining/triton_kernels/bench/bench_reduction.py:1)
- [triton_kernels/bench/bench_softmax.py](/Users/xavier/dev/cudatraining/triton_kernels/bench/bench_softmax.py:1)
- [triton_kernels/bench/bench_matmul.py](/Users/xavier/dev/cudatraining/triton_kernels/bench/bench_matmul.py:1)
- [triton_kernels/bench/bench_flash_attention.py](/Users/xavier/dev/cudatraining/triton_kernels/bench/bench_flash_attention.py:1)

GCP runner (5 파일):
- [scripts/gcp_run_lesson08_phase0.sh](/Users/xavier/dev/cudatraining/scripts/gcp_run_lesson08_phase0.sh:1) (smoke)
- [scripts/gcp_run_lesson08_phase1.sh](/Users/xavier/dev/cudatraining/scripts/gcp_run_lesson08_phase1.sh:1) (reduction)
- [scripts/gcp_run_lesson08_phase2.sh](/Users/xavier/dev/cudatraining/scripts/gcp_run_lesson08_phase2.sh:1) (softmax)
- [scripts/gcp_run_lesson08_phase3.sh](/Users/xavier/dev/cudatraining/scripts/gcp_run_lesson08_phase3.sh:1) (matmul)
- [scripts/gcp_run_lesson08_phase4.sh](/Users/xavier/dev/cudatraining/scripts/gcp_run_lesson08_phase4.sh:1) (FA)

벤치 CSV (L4 sm_89):
- [results/remote/reduction_3way.csv](/Users/xavier/dev/cudatraining/results/remote/reduction_3way.csv:1), [reduction_triton_sweep.csv](/Users/xavier/dev/cudatraining/results/remote/reduction_triton_sweep.csv:1)
- [results/remote/softmax_4way.csv](/Users/xavier/dev/cudatraining/results/remote/softmax_4way.csv:1), [softmax_triton_sweep.csv](/Users/xavier/dev/cudatraining/results/remote/softmax_triton_sweep.csv:1)
- [results/remote/matmul_3way_fp32.csv](/Users/xavier/dev/cudatraining/results/remote/matmul_3way_fp32.csv:1), [matmul_3way_fp16.csv](/Users/xavier/dev/cudatraining/results/remote/matmul_3way_fp16.csv:1)
- [results/remote/flash_attention_4way.csv](/Users/xavier/dev/cudatraining/results/remote/flash_attention_4way.csv:1)

문서:
- [docs/blog_draft_triton_vs_cuda.md](/Users/xavier/dev/cudatraining/docs/blog_draft_triton_vs_cuda.md:1) — 블로그 초안
- 이 파일 — 핸드오프

## 3. 핵심 숫자 (L4 sm_89)

### 3.1 메모리 바운드 (HBM 300 GB/s peak)

**Reduction, 67M fp32 (~256 MB, L2 overflow)**:

| | ms | GB/s | HBM % |
|---|---:|---:|---:|
| CUDA v4_shuffle | 1.039 | 258 | 86% |
| torch.sum | 1.056 | 254 | 85% |
| **Triton (autotune)** | 1.097 | **245** | **82%** |

**Softmax, 4096×4096 fp32**:

| | ms | GB/s |
|---|---:|---:|
| CUDA v2_fused | 0.565 | 237 |
| torch.softmax | 0.560 | 240 |
| **Triton (autotune)** | 0.608 | **221** |

**→ HBM 바운드는 세 접근이 10 % 안쪽에서 비김. Triton 이 손수 CUDA 의 93-95 %.**

### 3.2 컴퓨트 바운드 (matmul, TF32/fp16 TC)

**FP32 4096³ (TF32 TC peak 121 TFLOPS)**:

| | ms | TFLOPS | peak % |
|---|---:|---:|---:|
| CUDA v3 (FP32 FMA, no TC) | 35.6 | 3.9 | 3% |
| torch.matmul (cuBLAS + TF32) | 5.34 | 25.8 | 21% |
| **Triton (autotune)** | **4.75** | **28.9** | **24%** |

**FP16 4096³ (fp16 TC peak 242 TFLOPS)**:

| | ms | TFLOPS | peak % |
|---|---:|---:|---:|
| CUDA v4 (WMMA) | 7.42 | 18.5 | 8% |
| torch.matmul (cuBLAS) | 2.65 | 51.8 | 21% |
| **Triton (autotune)** | **2.54** | **54.0** | **22%** |

**→ 4096³ 에서 Triton 이 cuBLAS 를 근소하게 이김 (fp32 +12 %, fp16 +4 %). 우리 WMMA 대비 2.9 x.**

### 3.3 Flash Attention (head_dim = 64, best_ms)

| N | CUDA FA v1 fp32 | Triton fp32 | Triton fp16 | torch SDPA fp32 | torch SDPA fp16 |
|---:|---:|---:|---:|---:|---:|
| 1024 | 0.324 | 0.148 | 0.122 | 0.156 | 0.076 |
| 2048 | 0.638 | 0.196 | 0.138 | 0.279 | 0.076 |
| 4096 | 1.256 | 0.358 | 0.207 | 0.755 | 0.127 |
| 8192 | **3.045** | **1.118** | **0.496** | 2.633 | **0.394** |

**핵심 비율 (N=8192)**:
- **Triton fp16 vs 우리 CUDA FA v1 fp32 → 6.14 x 빠름**
- **Triton fp32 vs torch SDPA fp32 → 2.35 x 빠름** (torch 의 fp32 경로는 L4 에서 FA 안 타고 naive 로 떨어짐)
- **Triton fp16 vs torch SDPA fp16 (cuDNN FA-2) → 0.79 x** (우리가 25 % 느림)

**→ ~100 줄 Triton 이 cuDNN FA-2 의 79 % 속도, 동시에 우리 레슨 6 CUDA FA 를 6 배 차이로 압도.** 이게 Tri Dao 가 FA-2 를 Triton 으로 쓴 이유.

### 3.4 Autotune 이 고른 config (L4 참고용)

| kernel × shape | autotune choice | note |
|---|---|---|
| reduction 67M | `BS=512, nw=4` | 수동 sweep 최적과 0.3 % 차이 |
| softmax 4096² | `nw=8` | BLOCK_SIZE = 4096 (고정, next_pow2) |
| matmul fp32 4096³ | `BM=128, BN=128, BK=32, G=8, nw=4, ns=4` | 수동 스윕 불필요 |
| matmul fp16 4096³ | `BM=128, BN=256, BK=64, G=8, nw=8, ns=3` | fp16 이 더 큰 BN 선호 |
| FA fp16 N=8192 | `BM=128, BN=64, nw=8, ns=2` | 작은 N 은 BM=BN=64 로 수렴 |

## 4. 네 가지 교훈

### (a) **Triton 한 줄 = CUDA 수십 줄** 이 네 번 반복됐다

1. `tl.sum(x, axis=0)` 이 레슨 3 의 warp shuffle boilerplate (~15 줄 + sdata, `__shfl_down_sync`) 전체를 대체.
2. `tl.dot(a, b)` 가 레슨 5 의 WMMA fragment 선언 + `load_matrix_sync` + `mma_sync` (~50 줄) 를 대체. **입력 dtype 으로 TF32 TC vs fp16 TC 자동 선택.**
3. Grouped program ID swizzling 9 줄 (L2 cache 히트율을 위해 블록 순회를 비-row-major 로) 이 CUDA 에선 짜기 자체가 고역.
4. Flash Attention 의 online softmax 가 `tl.max + tl.maximum + tl.exp + tl.sum` 로 **그대로** 풀림. 레슨 6 에서 직접 짠 ~50 줄이 ~15 줄로 축약.

### (b) **Autotune 이 3 % 안쪽 최적화**. 단, footgun 동반

`@triton.autotune(configs=[...], key=["n_elements"])` 만 붙이면 자동 튜닝. 하지만:

- **Autotune 이 sweep 중에 output buffer 에 stale write** 를 남김. reduction 의 partial sum 이 오염돼서 correctness 가 rel_err = 1.75 로 터짐. 해결: `reset_to_zero=["partial_ptr"]` 인자 + 호출 후 `best_config.kwargs["BLOCK_SIZE"]` 로 slicing.
- **수동 sweep 대비 3 % 안쪽에서 수렴.** 이론 최적은 아니어도 실무적으로 충분.

### (c) **추상화의 비용은 "작은 커널" 에만 집중**

네 구간:

| 구간 | Triton vs CUDA | 원인 |
|---|---|---|
| 작은 N (< 4MB) | 3-12 x 뒤짐 | **Launch overhead floor ~50-100 µs** (Python → autotune 캐시 → JIT → cuLaunchKernel) |
| HBM 바운드 중간 | 95 % | 거의 없음 |
| HBM 바운드 큰 | 동률 | 없음 |
| 컴퓨트 바운드 큰 matmul/FA | **이김** | autotune 이 사람보다 나은 config 를 고름 |

**실무 해석**: Transformer layer 하나가 ≥1 ms 면 100 µs overhead 는 10 % 미만 — 인내 가능. 하지만 element-wise 30 개를 각각 Triton 으로 런치하면 망함. **작은 연산은 PyTorch eager 나 `torch.compile` 이 나음.**

### (d) **TF32 footgun 으로 벤치가 거짓말함** (하마터면)

PyTorch 의 `torch.matmul(fp32)` 는 기본적으로 **TF32 TC 를 사용하지 않음.** Triton `tl.dot` 는 기본 TF32 사용. 그대로 비교하면 "Triton 이 torch 를 2 x 이김" 처럼 보임 — 실제론 둘이 다른 하드웨어 경로를 탐.

공정하게 측정하려면:
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

이걸 세팅한 뒤에야 두 쪽 다 TF32 TC 경로. 이 footgun 을 알기 전까지 Phase 3 결과가 2 x 거짓이었음.

## 5. 빌드 시스템 & 벤치 함정 기록

### 함정 1: Autotune stale write (Phase 1 에서 디버그)

증상: 2-pass reduction 의 correctness 가 `rel_err = 1.75` 로 실패.

원인: `@triton.autotune` 이 각 config 를 시도하면서 **같은 `partial_ptr` 버퍼에 다른 길이로 write**. 실제 실행은 best_config 로 이뤄지지만, 이전 trial 의 stale partial sum 이 unwritten slot 에 남아서 `.sum()` 에 포함됨.

해결 (두 조합):
```python
@triton.autotune(
    configs=AUTOTUNE_CONFIGS, key=["n_elements"],
    reset_to_zero=["partial_ptr"],          # (1) trial 간 초기화
)
@triton.jit
def reduce_sum_kernel_autotuned(...):
    ...

def triton_reduce_sum_autotuned(x):
    ...
    best_cfg = reduce_sum_kernel_autotuned.best_config
    block = best_cfg.kwargs["BLOCK_SIZE"]   # (2) 유효 prefix 계산
    num_programs = triton.cdiv(n, block)
    return partial[:num_programs].sum()
```

### 함정 2: `allow_tf32 = False` 기본값 (Phase 3)

위 §4(d). 두 줄로 해결:
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### 함정 3: CUDA 바이너리 CSV 컬럼명 불일치 (Phase 3)

`bench_matmul.py` 가 `tflops` 를 기대했지만 레슨 5 의 CUDA 바이너리는 `effective_tflops` 로 출력. fallback chain:
```python
if "effective_tflops" in row:
    tflops = float(row["effective_tflops"])
elif "tflops" in row:
    tflops = float(row["tflops"])
else:
    tflops = tflops_of(M, N, K, best_ms)
```

### 함정 4: L2 cache 효과 (벤치 해석)

4 MB 텐서 reduction 에서 1170 GB/s 가 나옴. L4 HBM peak 300 GB/s 라 당연히 이상. 원인: 4 MB 는 L4 의 48 MB L2 안에 들어감 → 측정값이 L2 밴드폭. **의미있는 HBM 비교는 L2 를 overflow 하는 크기 (≥ 128 MB) 에서 해야 함.** 67 M 원소 (256 MB) 에서 측정한 245 GB/s 가 진짜 HBM 숫자.

## 6. 구조 매핑 (레슨 1-8 의 층)

```
Python 모델 / 유저 (vLLM, 내 서비스)
          │
          │   torch.ops.mylib.flash_attention(q, k, v)       ← 레슨 7 이 뚫은 층
          ▼
torch dispatcher   (device / dtype / autograd 체크)
          │
          ├──── triton_kernels.triton_flash_attention(q, k, v)   ← 레슨 8 이 뚫은 층
          │     (JIT → autotune → PTX 생성)
          │
          ▼
C++ host wrapper   (tensor → raw ptr, output alloc, stream)
          │
          │   kernel<<<grid, block, 0, stream>>>(...)
          ▼
CUDA kernel        (flash_attention_v1)                       ← 레슨 6
          │
          ▼
Warp / thread      (shuffle reduce, tiled matmul)             ← 레슨 3, 4, 5
          │
          ▼
Memory hierarchy   (HBM ↔ L2 ↔ smem ↔ register)               ← 레슨 1, 2
```

**레슨 7 이후**: CUDA 커널이 `torch.ops.mylib.*` 로 Python 에 노출됨.
**레슨 8 이후**: **같은 층에 Triton 이 대안 경로로 추가.** Python 에서 두 방향 다 뚫림.

실무적으로: 작은/중간 커널은 Triton, 커스텀 PTX 최적화가 필요하면 CUDA (torch.ops 경로). vLLM 이 실제로 두 접근을 섞어 쓰는 이유.

## 7. "그래서 CUDA 왜 계속 배우나" 의 네 가지 답

측정 결과만 보면 "Triton 이 다 해주네" 싶지만:

1. **Triton 이 막히는 순간 CUDA 로 떨어짐**. 예: `tl.dot` 가 지원 안 하는 mma (Blackwell BF8/FP4), persistent kernel, async copy 세부 제어.
2. **Triton 이 낸 PTX 를 읽을 줄 알아야 성능 bug 추적**. `TRITON_CACHE_DIR` 에 `*.ptx`, `*.cubin` 이 떨어짐.
3. **vLLM, FlashAttention-3, Mamba 커널은 아직 CUDA**. 그 코드를 읽으려면 CUDA 가 모국어여야.
4. **"왜 느림?" 의 답이 bank conflict, register spill, occupancy 같은 CUDA 개념**. Triton 코드가 느릴 때도 결국 그 언어로 해석.

비유: **CUDA = 어셈블리, Triton = C**. 대부분은 C 로 짜고, 핫패스만 어셈블리로 남긴다.

## 8. 다음 단계 후보

### Next A: **Triton FA + causal + dropout** (레슨 8.5)

실제 LLM 학습에 쓰이는 FA 의 완성 형태. 지금 Triton FA 는 non-causal forward only. causal mask + dropout + (선택) backward 추가.

**장점**: 바로 쓸 수 있는 production-ready 커널. **단점**: 규모가 Phase 4 의 2-3 배.

### Next B: **vLLM PagedAttention Triton 포팅** (Phase 1 M3 첫 걸음)

vLLM 의 `paged_attention_v1` 커널 (~500 줄 CUDA) 을 Triton 으로 재작성 → PR 로 제안. 만약 성공하면 Phase 1 M3 목표 (vLLM 코드 기여) 의 실제 결과물.

**장점**: 오픈소스 기여 포트폴리오. **단점**: block_size / KV cache layout 이해 필요 → 사전에 vLLM 코드 스터디 1-2 주.

### Next C: **CUTLASS 주력 (레슨 9)**

Triton 의 상한 (persistent kernel, warp specialization, epilogue fusion) 을 넘는 구간. CUTLASS 3.x 가 공식 답.

**장점**: Triton 이 못 내는 성능 구간을 배움. **단점**: 학습 곡선이 레슨 1-8 을 모두 합친 것보다 가파름.

### Next D: **Blackwell olympics** (하드웨어 업그레이드)

B200 에서 같은 벤치 재실행 — FP8 TC, TMA (tensor memory accelerator), thread block cluster. L4 대비 5-10 x 예상.

**장점**: 최신 세대 HW 감. **단점**: GCP 에 B200 없음 → AWS P6 or 모 GPU 클라우드.

### 추천 순서

**B → A → C** 가 Phase 1 최종 목표 (M4 = 작은 오픈소스 PR 한 건) 와 가장 잘 정렬됨:
1. **B (vLLM 포팅)**: 실제 오픈소스 기여 기회 + 시장성.
2. **A (Causal FA)**: B 하면서 필요해짐 (PagedAttention 은 causal 이 default).
3. **C (CUTLASS)**: Phase 2 로 넘어가면서.

D (Blackwell) 는 Phase 1 예산 여유 있을 때 1 회성 실험.

## 9. VM 상태

**`cuda-l4-dev-lesson08`** — us-west4-a, `g2-standard-4`, SPOT, **RUNNING**.

유저 선호도 ([feedback_vm_lifecycle.md](/Users/xavier/.claude/projects/-Users-xavier-dev-cudatraining/memory/feedback_vm_lifecycle.md:1)): 레슨 사이에 **VM 살려두기**. 자동 삭제 금지.

이번 세션 종료 후 상태:
- 레포 복사본이 `~/cudatraining/` 에 있음 (최신 버전).
- Python 환경: torch 2.6 + triton 3.2 + CUDA 12.4. 재설치 불필요.
- CUDA 바이너리 (sm_89): `bin/reduction`, `bin/softmax`, `bin/matmul`, `bin/flash_attention` — 다음 세션에 바로 쓸 수 있음.
- Triton JIT 캐시: `~/.triton/cache/` 에 PTX 유지. 같은 커널 재호출 시 warmup 단축.

**권장**: 다음 세션까지 > 1 일 gap 이면 `gcloud compute instances stop cuda-l4-dev-lesson08 --zone us-west4-a` (SPOT 이라도 정지로 과금 줄임). 짧으면 그대로.

## 10. 한 줄 요약

레슨 1-6 의 4 개 CUDA 커널을 ~40-130 줄짜리 **Triton** 으로 재작성. L4 에서 측정: **큰 matmul/FA 는 cuBLAS 와 동률 또는 이김** (Triton fp16 matmul 54 TF, FA fp16 이 우리 CUDA 대비 6.1 x). **추상화의 비용은 '작은 커널' 에만 존재** — Python launch floor ~50 µs. CUDA 는 여전히 bug 추적과 Triton 이 막히는 구간의 도피처.
