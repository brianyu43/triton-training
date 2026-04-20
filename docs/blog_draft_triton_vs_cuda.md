# Triton vs CUDA: 추상화의 비용은 어디서 나타나는가

*~50 줄의 Python 이 5000 줄의 CUDA 를 대체할 수 있는가? L4 (sm_89) 에서 4 개 커널을 양쪽 다 구현해서 비교했다.*

기준 날짜: 2026-04-19  ·  GPU: NVIDIA L4 (Ada Lovelace, 24GB)  ·  CUDA 12.4 · PyTorch 2.6 · Triton 3.2

## 1. 실험 설계

같은 L4 VM (GCP `g2-standard-4`, SPOT) 에서 네 개 커널을 각각 **세 가지로** 구현:

1. **레슨 1-6 에서 쓴 CUDA** — smem 타일링, warp shuffle, WMMA Tensor Core 등 수작업 구현
2. **PyTorch 내장** — `.sum()`, `torch.softmax`, `torch.matmul`, `F.scaled_dot_product_attention`. 내부는 cuBLAS / cuDNN / FA-2.
3. **Triton** — 이 레슨에서 포팅한 ~40-100 줄 Python 커널

벤치는 **best-of-50 CUDA event timing**, warmup 10. 결과는 [results/remote/](../results/remote/) 에 CSV 로 전부 보관.

| 커널 | Triton 파일 | CUDA 파일 | 공식 비교 |
|---|---|---|---|
| reduction (sum) | [triton_kernels/reduction.py](../triton_kernels/reduction.py) | [src/reduction.cu](../src/reduction.cu) `v4_shuffle` | `torch.sum` |
| softmax (row-wise) | [triton_kernels/softmax.py](../triton_kernels/softmax.py) | [src/softmax.cu](../src/softmax.cu) `v2_fused` | `torch.softmax` |
| matmul | [triton_kernels/matmul.py](../triton_kernels/matmul.py) | [src/matmul.cu](../src/matmul.cu) `v3_register` / `v4_tensor` | `torch.matmul` (cuBLAS) |
| flash attention | [triton_kernels/flash_attention.py](../triton_kernels/flash_attention.py) | [src/flash_attention.cu](../src/flash_attention.cu) `v1` | `F.scaled_dot_product_attention` (FA-2) |

L4 스펙: HBM bandwidth 300 GB/s, FP32 FMA peak 30 TFLOPS, **TF32 TC 121 TFLOPS**, FP16 TC 242 TFLOPS.

## 2. 결과 요약표

### 2.1 메모리 바운드 — reduction, softmax

**Reduction (fp32 sum, 67M elements)**:

| | ms | GB/s | vs peak HBM |
|---|---:|---:|---:|
| 우리 CUDA v4_shuffle | 1.039 | 258 | 86% |
| torch.sum | 1.056 | 254 | 85% |
| **Triton** | 1.097 | **245** | **82%** |

**Softmax (fp32, 4096×4096)**:

| | ms | GB/s |
|---|---:|---:|
| 우리 CUDA v2_fused | 0.565 | 237 |
| torch.softmax | 0.560 | 240 |
| **Triton** | 0.608 | **221** |

**→ HBM 바운드 영역에선 세 접근이 10% 안쪽에서 비김.** Triton 이 CUDA 의 **93-95%** 속도. 이 구간은 "얼마나 HBM 을 빨리 빨아들이는가" 가 전부라 컴파일러가 사람을 크게 못 넘는다.

### 2.2 컴퓨트 바운드 — matmul

**FP32 (M=N=K=4096, TF32 TC)**:

| | ms | TFLOPS | vs peak 121 |
|---|---:|---:|---:|
| 우리 CUDA v3 (FMA, no TC) | 35.5 | 3.9 | 3% |
| torch.matmul (cuBLAS+TF32) | 5.34 | 25.8 | 21% |
| **Triton** | **4.75** | **28.9** | **24%** |

**FP16 (M=N=K=4096)**:

| | ms | TFLOPS | vs peak 242 |
|---|---:|---:|---:|
| 우리 CUDA v4 (WMMA) | 7.34 | 18.5 | 8% |
| torch.matmul (cuBLAS) | 2.65 | 51.8 | 21% |
| **Triton** | **2.54** | **54.0** | **22%** |

**→ 4096³ 에서 Triton 이 cuBLAS 를 근소하게 이긴다** (fp32 +12%, fp16 +4%). 동시에 우리 CUDA WMMA 의 **2.9x 속도**. 이게 Triton 이 유명해진 이유.

### 2.3 Flash Attention (head_dim=64)

| N | 우리 CUDA FA fp32 | Triton fp32 | Triton fp16 | torch SDPA fp32 | torch SDPA fp16 |
|---:|---:|---:|---:|---:|---:|
| 1024 | 0.324 | 0.148 | 0.122 | 0.156 | 0.076 |
| 2048 | 0.638 | 0.196 | 0.138 | 0.279 | 0.076 |
| 4096 | 1.256 | 0.358 | 0.207 | 0.755 | 0.127 |
| 8192 | **3.045** | **1.118** | **0.496** | 2.633 | **0.394** |

**주요 비율 (N=8192)**:
- Triton fp16 vs 우리 CUDA FA fp32 → **6.1x 빠름**
- Triton fp32 vs torch SDPA fp32 → **2.35x 빠름** (torch 의 fp32 경로가 L4 에서 FA 안 탐)
- Triton fp16 vs torch SDPA fp16 (FA-2) → **0.79x** (우리가 25% 느림)

**→ ~40 줄 Triton 이 cuDNN/cutlass FA-2 의 79% 속도.** 동일 알고리즘인 우리 레슨 6 CUDA 를 6x 차이로 눌러버린다.

## 3. "추상화의 비용" 은 어디에 있는가

상식적으로는 "고수준 DSL = 성능 손실" 일 거 같지만, 실제로 측정하면:

| 구간 | Triton 성능 | 비용 |
|---|---|---|
| 작은 N (< 4MB 대략) | 매우 느림 (3-12x 뒤짐) | **launch overhead floor 50-100 µs** |
| HBM 바운드, 중간 N | CUDA 의 93-95% | 거의 없음 |
| HBM 바운드, 큰 N | CUDA 와 동률 | 없음 |
| 컴퓨트 바운드, 큰 matmul/FA | **cuBLAS 와 동률 또는 살짝 이김** | **없음 / 음수** |

한 마디로: **추상화의 비용은 "작은 커널" 에 집중**된다. 큰 커널에서는 비용이 0 이거나 음수다 (autotune 이 사람보다 나은 config 를 고르니까).

### 왜 작은 커널에서만 비용이 나타나는가

Triton 커널을 호출하면 다음이 일어남:
1. Python 에서 `kernel[grid](args, ...)` 호출
2. autotune 캐시 조회 (첫 호출이면 sweep 후 선택)
3. Triton 의 JIT 캐시 조회
4. argument binding + launch 파라미터 계산
5. `cuLaunchKernel` 호출

이 **1-4 단계가 합쳐서 ~50-100 µs**. 4MB FP32 텐서의 reduction 은 HBM 이론 기준 약 13 µs 면 끝나는데 이 overhead 에 묻힘.

CUDA 는 훨씬 얇다: C++ 런타임 체크 + `cudaLaunchKernel` → ~8 µs. 그래서 작은 커널에서 5-10x 빨라 보이는 것.

**실무에서 중요한가?** 트랜스포머 레이어 하나가 ≥1 ms 라면 100 µs overhead 는 10% 미만. 그 정도면 인내할 만. Transformer 를 element-wise 서른 개로 쪼개서 각각 Triton 커널로 돌리면 망함.

## 4. 네 번의 "Triton 한 줄이 CUDA 수십 줄" 순간

### 4.1 `tl.sum(x, axis=0)` 이 warp shuffle 전체를 대체

레슨 3 의 `reduce_v4_shuffle` 은:
```cpp
for (int offset = 16; offset > 0; offset >>= 1) {
  local += __shfl_down_sync(0xFFFFFFFFu, local, offset);
}
if (lane == 0) sdata[wid] = local;
__syncthreads();
int num_warps = blockDim.x >> 5;
if (wid == 0) {
  float v = (tid < num_warps) ? sdata[lane] : 0.0f;
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
  }
  if (tid == 0) atomicAdd(out, v);
}
```

Triton:
```python
partial = tl.sum(x, axis=0)
tl.store(partial_ptr + pid, partial)
```

컴파일러가 **같은 SASS** 를 낸다 (검증됨: `TRITON_PRINT_PTX=1` 로 보면 `__shfl_down_sync` 유사 instruction 가 보임).

### 4.2 `tl.dot(q, tl.trans(k)) * scale` 이 WMMA boilerplate 를 대체

레슨 5 v4_tensor 의 matmul:
```cpp
fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag[2];
fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> b_frag[2];
fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[2][2];
// 이어서 load_matrix_sync + mma_sync 수십 줄
```

Triton:
```python
acc = tl.dot(q, tl.trans(k))
```

Triton 이 입력 dtype 을 보고:
- fp32 → TF32 TC (mma.m16n8k8)
- fp16 → fp16 TC (mma.m16n8k16)
를 자동 선택.

### 4.3 Grouped program ID — CUDA 로는 손으로 못 짜는 swizzling

Triton matmul 의 standard trick:
```python
pid = tl.program_id(axis=0)
num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

블록 그리드를 row-major 로 돌지 않고 **GROUP_SIZE_M 만큼 M 방향 행을 끼어서 순회** — L2 캐시 히트율을 높임. CUDA 에서 같은 패턴을 구현하려면 blockIdx 계산을 커널 맨 앞에 이렇게 풀어 써야 하고, 그 코드는 **읽기 힘들다**. Triton 에선 이게 표준 관용구.

### 4.4 Online softmax 가 Flash Attention 에서 그대로 통한다

레슨 6 에서 online softmax 를 직접 짰던 50 줄을 Triton 은 안에서 다 해준다:

```python
s = tl.dot(q, tl.trans(k)) * scale
m_ij = tl.max(s, axis=1)
m_new = tl.maximum(m_i, m_ij)
alpha = tl.exp(m_i - m_new)
p = tl.exp(s - m_new[:, None])
l_ij = tl.sum(p, axis=1)
l_new = alpha * l_i + l_ij
acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
```

CUDA 로 같은 걸 짜면 `__shfl_xor_sync`, `max_warp_reduce`, `online_merge_warp` 를 따로 구현해야 함.

## 5. Autotune 의 실제 가치

`@triton.autotune(configs=[...], key=[...])` 데코레이터만 달면 첫 호출 때 config 목록을 돌면서 가장 빠른 걸 고름. 우리 매 phase 에서 본 패턴:

**Reduction (67M 원소)**:
- 수동 sweep 최적: `BS=512, nw=4` → 1.097 ms
- autotune 고름: `BS=2048, nw=8` → 1.100 ms (0.3% 느림)

**Matmul (4096³ fp16)**: autotune 이 고른 게 가장 빠름 (`BM=128, BN=256, BK=64, nw=8, ns=3`)

**FA (N=8192)**: autotune 이 `BM=128, BN=64, nw=8, ns=2` 로 수렴. 작은 N 에선 `BM=BN=64, nw=4, ns=3` 유지.

**교훈**:
- autotune 이 "이론적 최적" 은 안 줘도 "실무적으로 거의 최적" 은 준다.
- **수동 sweep 대비 3% 안쪽에서 converge**.
- 새 GPU 로 넘어갈 때도 같은 decorator 가 자동으로 re-tune. CUDA 로 짰으면 새 GPU 마다 block size 를 수동 재조정해야 함.

**footgun 하나**: `@triton.autotune` 이 출력 텐서에 config 별로 중첩 write 를 해서 결과가 오염될 수 있음 (우리가 Phase 1 에서 디버그). 해결: `reset_to_zero=["output_ptr"]` 인자 + best_config 이후 slicing.

## 6. 그럼 CUDA 는 왜 계속 배우나

측정을 끝내고 "Triton 이 다 해주는데 CUDA 왜 배움?" 소리가 나올 만한데 — 그러지 말아야 할 이유:

1. **Triton 이 막히는 순간에 CUDA 로 떨어진다**. `tl.dot` 가 지원 안 하는 mma instruction (예: BF8/FP4 mma on Blackwell), persistent kernel 로 autotune 못 하는 맞춤 스케줄링, async copy 세부 제어 등.
2. **Triton 이 생성한 PTX 를 읽을 줄 알아야 디버그**. `TRITON_CACHE_DIR` 밑에 `*.ptx`, `*.cubin` 이 떨어지고, 이걸 읽으려면 PTX / SASS 지식 필요.
3. **vLLM, FlashAttention-3, Mamba 등은 아직 CUDA 기반**. 그 코드 읽으려면 CUDA 가 모국어여야 함.
4. **성능 bug 추적**. Triton 코드가 느릴 때 "왜 느림?" 답을 찾으려면 smem bank conflict, register spill, occupancy 같은 CUDA 개념으로 돌아감.

**적정 표현**: CUDA 는 "어셈블리", Triton 은 "C". 대부분은 C 로 짜고, 핫패스 어셈블리만 남긴다.

## 7. 벤치 실수 모음 (미래 자산)

### 7.1 `allow_tf32` footgun

PyTorch 의 `torch.matmul(fp32)` 는 기본적으로 **TF32 Tensor Core 를 사용하지 않는다**. Triton 은 `tl.dot` 이 기본적으로 TF32 사용. 이대로 비교하면 torch 가 2x 느려 보이지만 실상은 둘이 다른 경로. 
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```
를 먼저 세팅한 뒤 비교해야 공정.

### 7.2 Autotune 이 중간 출력에 stale write 남김

`@triton.autotune` 이 sweep 중에 각 config 를 같은 output 버퍼에 써 버린다. 우리의 2-pass reduction 에선 이 stale partial sum 이 `.sum()` 에 포함되어 결과가 망가짐. 해결:
- `reset_to_zero=["partial_ptr"]` 로 trial 간 초기화
- 호출 후 `best_config.kwargs` 로 유효 prefix slicing

### 7.3 L2 캐시 효과 — 작은 N 의 밴드폭이 peak 을 넘는다

4MB 텐서 reduction 에서 1170 GB/s 가 나옴. L4 HBM 은 300 GB/s 라 당연히 이상. 원인: 4MB 는 L4 의 L2 cache (48MB) 안에 들어감. warmup 10 회 후엔 데이터가 전부 L2. **작은 텐서의 GB/s 는 "HBM 밴드폭" 이 아니라 "L2 밴드폭"**. 의미있는 비교는 L2 를 overflow 시키는 크기에서.

### 7.4 Launch overhead floor 가 실제 측정값을 왜곡

Triton 은 ~50-100 µs launch floor 가 있어서, 실제 연산이 10 µs 인 커널은 100 µs 로 측정된다. 벤치 CSV 의 GB/s 가 비현실적으로 낮다면 overhead 구간.

## 8. 다음에 뭐할 것인가

이 레슨의 부산물로 생긴 것들:
- `triton_kernels/` — 4 개 kernel + 4 개 bench
- `scripts/gcp_run_lesson08_phaseN.sh` — 재현 가능한 벤치 파이프라인
- 이 블로그 초안 — 나중에 정리해서 publish

**다음 단계 후보**:
1. **Triton FA + causal mask + dropout** — 실제 학습에 쓰이는 FA 의 완성 형태
2. **vLLM PagedAttention Triton 포팅** — 이 작업이 성공하면 Phase 1 M3 (vLLM 코드 기여) 의 첫걸음
3. **Blackwell (B200) 으로 olympics** — FP8 tensor cores 가 있는 GPU 에서 같은 벤치. L4 대비 ~5-10x 예상.
4. **레슨 9: CUTLASS 주력** — 더 낮은 층 (예: `epilogue fusion`, `warp specialization`). Triton 의 상한을 넘는 구간.

---

## Appendix: 전체 CSV 매트릭스

| Phase | CSV 파일 | 핵심 컬럼 |
|---|---|---|
| 1 | `results/remote/reduction_3way.csv` | triton_best_ms, triton_best_cfg, triton_autotune_cfg, torch_sum_ms, cuda_v4_ms, *_gbps |
| 1 (sweep) | `results/remote/reduction_triton_sweep.csv` | n, BLOCK_SIZE, num_warps, ms |
| 2 | `results/remote/softmax_4way.csv` | triton_best_ms, triton_best_cfg, triton_autotune_cfg, torch_softmax_ms, cuda_v2_ms, *_gbps |
| 2 (sweep) | `results/remote/softmax_triton_sweep.csv` | M, N, num_warps, ms |
| 3 (fp32) | `results/remote/matmul_3way_fp32.csv` | triton_ms, triton_cfg, triton_tflops, torch_ms, cuda_v3_ms |
| 3 (fp16) | `results/remote/matmul_3way_fp16.csv` | 동일 + cuda_v4 |
| 4 | `results/remote/flash_attention_4way.csv` | N, triton_fp32/fp16_ms, torch_sdpa_fp32/fp16_ms, cuda_flash_ms + autotune cfg |
