# Lesson 07 Handoff — PyTorch Custom Op Bridge

기준 날짜: `2026-04-19`

주제: **CUDA 커널을 `torch.ops.mylib.*` 로 등록. vLLM / FlashAttention 이 쓰는 공식 패턴.**

## 1. 이번 세션에서 한 일

### CUDA ↔ Python 브릿지 완성

`extension/csrc/flash_attention_ext.cu` — **한 파일에 전부**:
- 레슨 6 의 naive 3 커널 + flash_attention_v1 커널 (그대로 복사)
- C++ host wrapper 2 개:
  - `flash_attention_forward(Q, K, V)` — tensor 체크 → output alloc → stream-aware 런치
  - `naive_attention_forward(Q, K, V)` — S/P 버퍼 alloc 포함
- `TORCH_LIBRARY(mylib, m)` 로 op 스키마 선언 (`Tensor q, Tensor k, Tensor v -> Tensor`)
- `TORCH_LIBRARY_IMPL(mylib, CUDA, m)` 로 CUDA backend 구현 연결
- `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}` 빈 모듈 — `import` 가능하게 하는 관례

### 빌드 시스템

`extension/setup.py` — `torch.utils.cpp_extension.CUDAExtension`. `TORCH_CUDA_ARCH_LIST="7.5"` 로 T4 타겟.

### 테스트 + 벤치

`extension/python/test_correctness.py`:
- N ∈ {128, 512, 1024, 2048} × {naive, flash} = 8 correctness check (전부 통과)
- Guard 시연 3 종:
  - non-contiguous 입력 → `"Q must be contiguous"` 레이즈
  - FP16 입력 → `"Q must be float32"`
  - CPU 텐서 → dispatcher 가 `"not available for CPU backend"` 레이즈 (자동)

`extension/bench/bench_ops.py`:
- 우리 naive / 우리 flash / `F.scaled_dot_product_attention` 3-way 비교
- CUDA Event 기반 best-of-50 타이밍

### GCP 자동화

`scripts/gcp_run_lesson07.sh`:
- pip + PyTorch (cu121 wheels) 자동 설치
- `CUDA_HOME=/usr/local/cuda` 세팅
- `setup.py build_ext --inplace` 로 `mylib_ext.cpython-310-*.so` 생성
- test + bench 순차 실행 → CSV 회수

## 2. 산출물

- [extension/csrc/flash_attention_ext.cu](/Users/xavier/dev/cudatraining/extension/csrc/flash_attention_ext.cu:1) — 약 330 줄
- [extension/setup.py](/Users/xavier/dev/cudatraining/extension/setup.py:1)
- [extension/python/test_correctness.py](/Users/xavier/dev/cudatraining/extension/python/test_correctness.py:1)
- [extension/bench/bench_ops.py](/Users/xavier/dev/cudatraining/extension/bench/bench_ops.py:1)
- [scripts/gcp_run_lesson07.sh](/Users/xavier/dev/cudatraining/scripts/gcp_run_lesson07.sh:1)
- [results/remote/flash_attention_torchop_t4.csv](/Users/xavier/dev/cudatraining/results/remote/flash_attention_torchop_t4.csv:1)
- [results/lesson07-run-20260419-185624.log](/Users/xavier/dev/cudatraining/results/lesson07-run-20260419-185624.log:1)

## 3. 핵심 숫자

### Correctness (vs `F.scaled_dot_product_attention`)

| N | naive abs err | flash abs err |
|---:|---:|---:|
| 128  | 3.3e-07 | 2.5e-07 |
| 512  | 5.4e-07 | 4.3e-07 |
| 1024 | 4.0e-07 | 2.8e-07 |
| 2048 | 3.7e-07 | 3.7e-07 |

**FP32 machine epsilon (~1.2e-7) 의 약 3-5 배**. SDPA 가 이미 FP32 rounding 의 한계라서 우리 결과와 둘 다 "정답" 범위.

### Speed (best ms, T4, FP32, d=64, single head)

| N | ours_naive | ours_flash | **torch SDPA** | flash / sdpa |
|---:|---:|---:|---:|---:|
| 512  | 0.475 | 0.734 | **0.262** | 0.36x |
| 1024 | 0.799 | 1.309 | **0.429** | 0.33x |
| 2048 | 3.086 | 1.253 | **0.428** | 0.34x |
| 4096 | —     | 2.498 | **1.374** | 0.55x |

우리 flash 는 SDPA 대비 **0.33-0.55x 속도**. SDPA 는 T4 FP32 에서 cuDNN 의 튜닝된 FMA 패스로 도달.

**N 커질수록 격차 좁아짐** (4096 에서 0.55x). SDPA 도 결국 N² 연산량에 묶여서 우리 커널과 점점 비슷한 bound 에 도달.

## 4. 네 가지 교훈

### (a) **한 파일 ~50 줄** 이 PyTorch 세계로의 전환 전부

```cpp
check_qkv(Q, "Q");                           // shape/device/dtype/contig
auto O = torch::empty({N, d}, Q.options());  // PyTorch 가 alloc
auto stream = at::cuda::getCurrentCUDAStream();
my_kernel<<<grid, block, 0, stream>>>(Q.data_ptr<float>(), ..., O.data_ptr<float>(), ...);
return O;
```

이 5 줄이 `src/flash_attention.cu` 의 `main()` 수백 줄 (CLI, 메모리 alloc, CSV 출력, CPU ref 검증) 을 완전히 대체. **이게 production 패턴.**

### (b) **Stream-aware 런치** 가 단일 최고 중요 디테일

`at::cuda::getCurrentCUDAStream()` 을 네 번째 런치 인자에 안 주면, 우리 커널은 default stream 으로 런치되고 PyTorch 는 이미 다른 stream 을 쓰고 있을 수 있음 → **침묵의 race condition**. 테스트에서 크래시 안 나지만 결과 비결정적.

**vLLM 의 PagedAttention 커널 런치 라인도 정확히 이 패턴.**

### (c) **TORCH_LIBRARY 가 dispatcher 와 통합** — CPU 텐서 가드 "자동" 발생

우리 코드엔 `TORCH_CHECK(Q.is_cuda())` 가 있지만, 실제로 CPU 텐서 넘기면 그 체크 전에 **dispatcher 가 먼저 에러 레이즈**:
```
Could not run 'mylib::flash_attention' with arguments from the 'CPU' backend
```

왜? `TORCH_LIBRARY_IMPL(mylib, CUDA, m)` 로 CUDA only 구현 등록 → dispatcher 가 다른 device 면 우리 함수를 호출조차 안 함. 이게 PyTorch op 시스템의 대단히 깔끔한 부분. **autograd, dtype promotion, device dispatch 가 하나의 파이프에 엮임.**

### (d) **우리 vs cuDNN 3x 격차** 는 앞으로의 선이자 현실

T4 FP32 에서 SDPA 가 3x 빠른 이유 (추정):
1. **cuDNN 가 상수 인자 한계 튜닝**: register allocation, instruction scheduling, loop unrolling 을 NVIDIA 엔지니어가 사람 손으로
2. **warp-level matmul 패턴** (WMMA 는 FP16 전용이지만 FP32 에서도 warp-specialized 커널 있음)
3. **Smem bank conflict 회피** 레이아웃
4. **우리 Br=64 / d=64** 에서 register pressure 가 높음 — 스필 가능성

Triton / CUTLASS 레벨로 올라가면 이 격차를 절반으로 줄일 수 있음. FA-2 가 cuDNN 을 넘어선 건 Ampere+ 에서. T4 는 근본적으로 Tensor Core FP32 가 없는 구세대.

**교훈**: "내 커널 쓸래" vs "내장 라이브러리 쓸래" 는 항상 trade-off. 우리 자리는 **SOTA 를 쫓는 게 아니라, 내장에 빠진 기능 (novel op, custom sparsity pattern) 을 채우는 곳**.

## 5. 빌드 시스템 함정 기록

세션 중에 겪은 두 실수 — 미래 자산:

### 함정 1: `import mylib_ext` 가 `PyInit_mylib_ext not defined` 로 실패

원인: TORCH_LIBRARY 만 썼음. 이 매크로는 pybind11 의 `PyInit_*` 심볼을 안 만듬.
해결: 빈 `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}` 추가. **모든 PyTorch 확장 튜토리얼이 조용히 이걸 포함하는 이유.**

### 함정 2: `/usr/local/cuda/bin/nvcc` 가 PATH 에 없어서 `setup.py` 실패

해결: `export CUDA_HOME=/usr/local/cuda; export PATH=$CUDA_HOME/bin:$PATH`. PyTorch 의 CUDAExtension 이 nvcc 를 찾는 경로.

### 함정 3 (예고): CUDA 12.9 vs PyTorch cu121

VM 의 CUDA toolkit 은 12.9, PyTorch 는 12.1 로 빌드된 wheel. mismatch warning 나지만 실제론 돌아감 (minor version compat). **프로덕션에선 일치시키는 게 안전.**

## 6. 구조 매핑 (레슨 1-7 의 층)

```
Python 모델 / 유저 (vLLM, 내 서비스)
          │
          │   torch.ops.mylib.flash_attention(q, k, v)     ← 레슨 7 이 뚫은 층
          ▼
torch dispatcher   (device / dtype / autograd 체크)
          │
          ▼
C++ host wrapper   (tensor → raw ptr, output alloc, stream)
          │
          │   kernel<<<grid, block, 0, stream>>>(...)
          ▼
CUDA kernel        (flash_attention_v1)                  ← 레슨 6
          │
          ▼
Warp / thread      (shuffle reduce, tiled matmul)        ← 레슨 3, 4, 5
          │
          ▼
Memory hierarchy   (HBM ↔ L2 ↔ smem ↔ register)          ← 레슨 1, 2
```

**레슨 7 이전**: 우리 커널은 standalone 바이너리. **레슨 7 이후**: PyTorch 모델에 꽂히는 부품.

## 7. 이제 vLLM 코드가 읽힐 것

테스트: `vllm/csrc/attention/attention_kernels.cu` 열어서:
- 파일 위쪽 `#include <torch/extension.h>` — 익숙함
- `TORCH_CHECK(query.is_contiguous(), ...)` — 우리가 쓴 것과 똑같음
- `torch::Tensor paged_attention_v1(...)` host wrapper — 우리 패턴과 동일
- `paged_attention_v1_kernel<<<grid, block, shared_mem_size, stream>>>(...)` — stream-aware 런치

**"아, 이거구나"**. CUDA 학습 Phase 1 의 소목표 달성.

## 8. 다음 단계 후보

### Next A: **Triton 포팅** (Phase 1 M1-2)

지금 환경이 세팅됨 (VM + PyTorch + 빌드 체인). Triton 설치만 추가. 6 개 커널 (reduction / matmul / softmax / flash) 을 Triton 으로 재작성 + 블로그.

**가장 자연스러운 다음 단계**. 빌드 시스템 재사용.

### Next B: FA 확장 — causal mask + FP16 Tensor Cores

여기서 쌓은 PyTorch op 프레임에 FA v2 방향 기능 추가. 실제 LLM 과 가까워짐.

### Next C: vLLM 코드 리딩 (**새 흐름**)

더 이상 레슨이 아님. Phase 1 M2 의 활동. `vllm/csrc/` 의 주요 커널 3-5 개를 읽고 한 줄씩 주석 블로그.

### 추천

**C (vLLM 리딩) 를 먼저 간단히** 한 번 — §7 의 가설이 진짜 맞는지 체크. 30 분 이내. 맞다면 레슨 7 의 가치가 검증됨.

그 다음 **A (Triton 포팅)** 로 장기 궤도 복귀.

## 9. VM 상태

`cuda-t4-dev-lesson02`: RUNNING. 사용 끝났으면 stop.

## 10. 한 줄 요약

레슨 6 의 커널을 **~50 줄의 host wrapper** 로 감싸 `torch.ops.mylib.flash_attention` 로 부름. **8/8 correctness + 3/3 guard + PyTorch 내장 대비 0.33-0.55x 속도**. 이 순간부터 `vllm/csrc/*.cu` 는 낯선 코드가 아님.
