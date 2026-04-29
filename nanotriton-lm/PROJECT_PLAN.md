# NanoTriton-LM 프로젝트 계획서

> 작은 GPT 계열 언어모델을 직접 학습시키면서, Transformer의 핵심 병목 연산을 PyTorch reference에서 직접 작성한 Triton 커널로 단계적으로 교체하는 시스템 엔지니어링 프로젝트.

작성 기준일: 2026-04-28
프로젝트 가칭: `NanoTriton-LM`

---

## 1. 프로젝트의 진짜 목표

이 프로젝트의 목표는 “성능 좋은 LLM을 만드는 것”이 아니다. 목표는 아래 세 가지를 한 번에 증명하는 것이다.

1. 작은 decoder-only Transformer를 밑바닥부터 구현하고 학습시킬 수 있다.
2. Transformer 내부의 핵심 연산을 Triton 커스텀 커널로 직접 구현할 수 있다.
3. 커널 교체가 정확도, 속도, 메모리, kernel launch 수, profiler trace에 어떤 영향을 주는지 측정하고 설명할 수 있다.

따라서 최종 결과물은 단순한 학습 코드가 아니라, 다음을 포함하는 포트폴리오형 리포지토리여야 한다.

- 순수 PyTorch baseline 모델
- Triton 커널 기반 모델
- 커널별 correctness test
- 커널별 microbenchmark
- end-to-end training benchmark
- PyTorch Profiler / Nsight Systems 기반 분석 리포트
- “어떤 커널은 빨라졌고, 어떤 커널은 왜 PyTorch/cuBLAS보다 느린가”에 대한 정직한 분석

핵심 문장은 이렇게 잡으면 좋다.

> “I built and trained a tiny GPT-style language model, replacing major Transformer operations with custom Triton kernels, and validated each kernel through numerical tests, gradient checks, microbenchmarks, and end-to-end profiling.”

---

## 2. 프로젝트 범위

### 2.1 Core project scope

이 프로젝트는 레슨의 연장이 아니라 독립 포트폴리오 프로젝트다. 따라서 범위는 작게 접기보다, 안정적인 기준선에서 출발해 충분한 시간을 들여 Transformer training stack 의 주요 병목을 하나씩 교체하는 방향으로 잡는다. 다만 모든 것을 한 번에 Triton으로 대체하지는 않는다. `reference → partial Triton → mostly Triton` 순서로 가야 디버깅 가능성이 유지된다.

- 모델: 3M~20M parameter 수준의 decoder-only GPT
- 데이터셋: Tiny Shakespeare 또는 TinyStories subset
- 직접 구현할 Triton 커널:
  - RMSNorm forward/backward
  - Fused SwiGLU forward/backward
  - 기본 matmul 또는 Linear forward/backward
  - QKV projection 관련 커널 또는 layout transform 커널
  - simplified causal FlashAttention forward
  - 가능하면 FlashAttention backward까지
- PyTorch에 남겨둘 수 있는 부분:
  - tokenizer
  - dataloader
  - optimizer, 특히 AdamW
  - checkpoint 저장/로드
  - 일부 baseline용 matmul
  - sampling/generation loop

최종 목표는 “가능한 한 많은 핵심 연산을 Triton으로 이해하고 대체한 작은 training stack” 이지만, 각 kernel 은 독립적으로 검증된 뒤 모델에 들어가야 한다.

### 2.2 Full project extensions

Core training path가 안정화된 뒤에 아래를 확장한다.

- FlashAttention backward 완성
- custom AdamW Triton kernel
- fused residual + RMSNorm
- fused Linear + activation
- RoPE kernel
- fused cross entropy
- persistent matmul 실험
- FP8/FP4 같은 저정밀 실험
- `torch.compile` baseline과 비교
- CUDA kernel 버전과 Triton 버전 비교

---

## 3. 성공 기준

이 프로젝트는 “코드가 돌아감”만으로 끝나면 안 된다. 아래 기준을 통과해야 한다.

### 3.1 Correctness 기준

각 Triton op마다 다음을 확인한다.

- forward output이 PyTorch reference와 일치한다.
- backward gradient가 PyTorch autograd reference와 일치한다.
- fp32에서는 매우 엄격하게, fp16/bf16에서는 현실적인 tolerance로 비교한다.
- 다양한 shape에 대해 테스트한다.
- edge case를 포함한다.
  - sequence length가 block size로 나누어떨어지지 않는 경우
  - batch size가 작은 경우
  - hidden size가 128, 256, 384 등으로 바뀌는 경우
  - causal mask가 있는 경우

권장 tolerance 예시:

| dtype | forward atol/rtol | backward atol/rtol | 비고 |
|---|---:|---:|---|
| fp32 | `1e-5 / 1e-5` | `1e-4 / 1e-4` | 작은 shape에서 기준 |
| fp16 | `1e-2 / 1e-2` | `2e-2 / 2e-2` | accumulation은 fp32 권장 |
| bf16 | `2e-2 / 2e-2` | `3e-2 / 3e-2` | GPU 지원 여부 확인 |

### 3.2 Training 기준

- PyTorch baseline이 정상적으로 loss를 낮춘다.
- Triton 커널을 하나씩 교체해도 loss curve가 baseline과 비슷하게 내려간다.
- 특정 커널 교체 후 loss가 튀면, 해당 커널의 forward/backward를 다시 검증한다.
- 최소 하나의 dataset에서 짧은 generation sample을 생성한다.

### 3.3 Performance 기준

커널별, 모델별로 다음을 측정한다.

- latency
- tokens/sec
- peak CUDA memory
- kernel launch count
- profiler trace에서 병목 위치
- 주요 op별 self CUDA time
- attention score matrix를 명시적으로 만들었는지 여부
- end-to-end step time

속도가 무조건 PyTorch보다 빨라야 하는 것은 아니다. 특히 GEMM은 vendor library가 매우 강력하기 때문에 초반 Triton matmul이 느려도 괜찮다. 중요한 것은 “왜 느린지/빠른지”를 profiler와 roofline 관점으로 설명하는 것이다.

---

## 4. 모델 설계

### 4.1 모델 계열

decoder-only GPT 구조를 사용한다.

```text
Token Embedding
+ Positional Embedding 또는 RoPE
↓
N × Transformer Block
  - RMSNorm 또는 LayerNorm
  - Causal Multi-Head Self-Attention
  - Residual
  - RMSNorm 또는 LayerNorm
  - MLP: SwiGLU or GELU FFN
  - Residual
↓
Final Norm
↓
LM Head
↓
Cross Entropy Loss
```

Triton 프로젝트에는 LayerNorm보다 RMSNorm이 더 좋다. 이유는 수식이 약간 단순하고, LLaMA류 모델과도 잘 맞으며, mean subtraction이 없어서 backward 구현이 LayerNorm보다 덜 복잡하다.

### 4.2 모델 크기 후보

#### Tier 0: 디버깅용 초소형 모델

| 항목 | 값 |
|---|---:|
| parameter 수 | 약 1M~3M |
| layers | 2 |
| hidden size | 128 |
| heads | 4 |
| head dim | 32 |
| sequence length | 128 |
| FFN hidden | 384 또는 512 |
| dataset | Tiny Shakespeare char-level |

목적은 학습 성능이 아니라 디버깅이다. 커널이 잘못되면 이 모델에서도 loss가 바로 이상해진다.

#### Tier 1: Core 모델

| 항목 | 값 |
|---|---:|
| parameter 수 | 약 5M~15M |
| layers | 4 |
| hidden size | 256 |
| heads | 4 또는 8 |
| head dim | 64 또는 32 |
| sequence length | 256 |
| FFN hidden | 768 또는 1024 |
| dataset | Tiny Shakespeare, TinyStories subset |

이 크기가 가장 추천된다. 커널 테스트, end-to-end 학습, profiler 분석을 모두 하기에 부담이 적다.

#### Tier 2: 포트폴리오 데모 모델

| 항목 | 값 |
|---|---:|
| parameter 수 | 약 20M~50M |
| layers | 6~8 |
| hidden size | 384 또는 512 |
| heads | 6 또는 8 |
| head dim | 64 |
| sequence length | 512 |
| FFN hidden | 1536 또는 SwiGLU 기준 약 `8/3 × hidden` 계열 |
| dataset | TinyStories subset 또는 작은 OpenWebText류 subset |

이 모델은 최종 리포트용이다. 처음부터 여기에 들어가면 디버깅 비용이 너무 크다.

---

## 5. 데이터셋 계획

### 5.1 1차: Tiny Shakespeare

장점:

- 준비가 빠르다.
- char-level 모델로 tokenizer 복잡도를 제거할 수 있다.
- 데이터가 작아서 반복 실험이 쉽다.
- baseline loss curve가 빠르게 나온다.

단점:

- 실제 LLM tokenizer/word-level behavior와는 거리가 있다.
- 모델이 그럴듯한 영어 문장을 만드는지 보기에는 한계가 있다.

사용 목적:

- baseline 학습 확인
- 커널 교체 후 loss regression test
- CI-friendly test

### 5.2 2차: TinyStories subset

장점:

- 작은 모델이 문장다운 출력을 만들기에 좋다.
- 10M 이하 모델에서도 결과물이 시각적으로 그럴듯하게 보일 수 있다.
- 포트폴리오 데모에 좋다.

단점:

- tokenizer와 dataset pipeline을 더 신경 써야 한다.
- 전체 데이터셋은 생각보다 클 수 있으므로 subset부터 써야 한다.

사용 목적:

- 최종 demo training
- generation sample
- model scaling 실험

---

## 6. 외부 GitHub reference 사용 원칙

문서에서 GitHub repository, model, tutorial implementation 을 언급하면 단순 링크로 끝내지 않고, 실제로 인터넷에서 가져와 재현 가능한 reference 로 남긴다.

원칙:

- GitHub source 는 `references/` 또는 `third_party/` 아래에 fetch 한다.
- fetch script 를 둔다. 예: `scripts/fetch_references.py` 또는 `scripts/fetch_nanogpt.sh`.
- 가져온 repository 는 commit hash 를 pin 한다.
- `references/MANIFEST.md` 에 source URL, commit, license, 사용 목적을 기록한다.
- reference code 는 그대로 복사해 “내 코드”처럼 섞지 않는다. 비교/학습/검증 기준으로 둔다.
- 실제 구현은 `nanotriton/` 아래에 새로 작성하고, 필요한 경우 reference 와 diff 하기 쉽게 작게 유지한다.

초기 reference 후보:

| reference | source | 사용 목적 |
|---|---|---|
| nanoGPT | `https://github.com/karpathy/nanoGPT` | minimal GPT training loop / config / generation reference |
| Triton tutorials | `https://github.com/triton-lang/triton` 또는 공식 docs | RMSNorm, matmul, fused attention 구현 패턴 참고 |
| FlashAttention | `https://github.com/Dao-AILab/flash-attention` | attention forward/backward 수식 및 benchmark 관점 참고 |

이 원칙의 목적은 두 가지다. 첫째, 인터넷 reference 를 실제로 가져와서 재현 가능한 비교 기준을 만든다. 둘째, 포트폴리오에서 attribution 이 깨끗하게 보이도록 한다.

---

## 7. 리포지토리 구조

```text
nanotriton-lm/
  README.md
  PROJECT_PLAN.md
  pyproject.toml
  requirements.txt

  configs/
    tiny_shakespeare_ref.yaml
    tiny_shakespeare_triton_rmsnorm.yaml
    tiny_shakespeare_triton_all.yaml
    tinystories_10m.yaml

  data/
    shakespeare_char/
      prepare.py
    tinystories/
      prepare.py

  nanotriton/
    __init__.py

    model_ref.py              # 순수 PyTorch GPT
    model_triton.py           # Triton op를 사용하는 GPT
    modules/
      norm.py                 # RMSNorm module wrapper
      mlp.py                  # SwiGLU / FFN module wrapper
      attention.py            # naive attention / flash attention wrapper
      linear.py               # Triton Linear wrapper
      block.py                # Transformer block

    kernels/
      __init__.py
      vector_add.py
      rmsnorm.py
      swiglu.py
      matmul.py
      linear.py
      qkv.py
      softmax.py
      flash_attn.py
      adamw.py                # stretch
      cross_entropy.py        # stretch

    autograd/
      rmsnorm_fn.py
      swiglu_fn.py
      linear_fn.py
      flash_attn_fn.py

    train.py
    generate.py
    checkpoint.py
    tokenizer.py
    utils.py

  tests/
    test_rmsnorm.py
    test_swiglu.py
    test_matmul.py
    test_linear.py
    test_attention.py
    test_flash_attn.py
    test_model_equivalence.py
    test_training_smoke.py

  benchmarks/
    bench_rmsnorm.py
    bench_swiglu.py
    bench_matmul.py
    bench_attention.py
    bench_e2e_train.py
    profile_train.py

  reports/
    figures/
    traces/
    benchmark_tables/
    final_report.md

  references/
    MANIFEST.md
    nanogpt/                # fetched reference, pinned commit
```

중요한 점은 `kernels/`, `autograd/`, `modules/`를 분리하는 것이다. Triton kernel 자체와 PyTorch autograd wrapper, 그리고 모델 모듈이 섞이면 디버깅이 어려워진다.

---

## 8. 개발 원칙

각 커널은 반드시 같은 순서로 개발한다.

```text
1. 수식 정의
2. PyTorch reference 구현
3. 작은 shape에 대한 손계산 또는 numpy 비교
4. Triton forward 구현
5. forward allclose test
6. backward 수식 유도
7. Triton backward 구현
8. backward gradient 비교
9. microbenchmark
10. 모델에 삽입
11. loss curve regression test
12. profiler 분석
```

이 순서를 건너뛰면 나중에 “수식이 틀렸는지, Triton pointer가 틀렸는지, dtype이 틀렸는지, 모델이 불안정한지”를 분리할 수 없다.

---

## 9. 단계별 로드맵

## Phase 0. 환경 세팅과 실험 기반 만들기

목표: “언제든 같은 실험을 다시 돌릴 수 있는 상태”를 만든다.

### 할 일

- CUDA GPU 환경 확인
- PyTorch, Triton, pytest, numpy, matplotlib 설치
- GPU 정보 기록
  - GPU 모델
  - CUDA version
  - PyTorch version
  - Triton version
  - compute capability
- random seed 고정
- benchmark helper 작성
- `torch.cuda.synchronize()`를 포함한 timing utility 작성
- profiler helper 작성
- `pytest` 기반 테스트 환경 구성

### 산출물

- `requirements.txt` 또는 `pyproject.toml`
- `scripts/env_report.py`
- `benchmarks/utils.py`
- `tests/test_sanity.py`

### 체크리스트

- [ ] `python -c "import torch, triton"` 통과
- [ ] CUDA tensor 생성 가능
- [ ] 간단한 Triton vector add 실행 가능
- [ ] pytest 실행 가능
- [ ] benchmark 결과가 JSON/CSV로 저장됨

---

## Phase 1. 순수 PyTorch baseline 만들기

목표: 모든 Triton 커널의 정답지가 될 모델을 만든다.

### 할 일

- nanoGPT 스타일의 최소 GPT 구현
- 모델 config dataclass 또는 YAML 구성
- dataset prepare script 작성
- training loop 작성
- validation loop 작성
- generation script 작성
- checkpoint 저장/로드
- loss curve logging

### Baseline 구조

```text
GPTConfig(
  vocab_size,
  block_size,
  n_layer,
  n_head,
  n_embd,
  dropout,
  norm_type="rmsnorm" 또는 "layernorm",
  mlp_type="swiglu" 또는 "gelu"
)
```

### 추천 baseline 명령어

```bash
python data/shakespeare_char/prepare.py
python -m nanotriton.train --config configs/tiny_shakespeare_ref.yaml
python -m nanotriton.generate --ckpt out/ref/checkpoint.pt --prompt "To be"
```

### 산출물

- `model_ref.py`
- `train.py`
- `generate.py`
- baseline loss curve
- baseline generation sample

### 성공 기준

- loss가 안정적으로 감소한다.
- checkpoint reload 후 generation이 된다.
- 같은 seed에서 거의 같은 loss curve가 나온다.
- `model_ref.py`가 Triton 없이 독립적으로 동작한다.

---

## Milestone 1. Reference Training Stack

첫 번째 실행 계획은 커널 구현이 아니라 기준선 구축이다. 이 단계의 목적은 이후 Triton 커널을 하나씩 교체할 때 “정확도, 학습 안정성, 속도”를 판정할 정답지를 만드는 것이다.

### 범위

- 프로젝트 scaffold 생성
- 외부 GitHub reference fetch 체계 생성
- Tiny Shakespeare char-level dataset 준비
- PyTorch GPT baseline 구현
- RMSNorm + SwiGLU 기반 Transformer block 사용
- training / validation / checkpoint / generation loop 구현
- benchmark timing helper 와 environment report 작성
- pytest smoke test 작성

### 파일 단위 산출물

```text
pyproject.toml
configs/tiny_shakespeare_ref.yaml
data/shakespeare_char/prepare.py
nanotriton/config.py
nanotriton/model_ref.py
nanotriton/train.py
nanotriton/generate.py
nanotriton/tokenizer.py
nanotriton/utils.py
benchmarks/utils.py
scripts/env_report.py
scripts/fetch_references.py
references/MANIFEST.md
tests/test_sanity.py
tests/test_model_ref.py
```

### Definition of Done

- `python scripts/fetch_references.py --name nanogpt` 로 nanoGPT reference 를 가져오고 commit hash 를 기록한다.
- `python data/shakespeare_char/prepare.py` 가 dataset cache 를 만든다.
- `python -m nanotriton.train --config configs/tiny_shakespeare_ref.yaml` 로 500~1000 step 학습이 가능하다.
- train/val loss 가 저장되고 감소 추세를 보인다.
- checkpoint reload 후 `python -m nanotriton.generate ...` 가 동작한다.
- `pytest tests -q` 가 통과한다.
- `python scripts/env_report.py` 가 GPU / CUDA / PyTorch / Triton 정보를 출력한다.

---

## Phase 2. Triton 기본기: vector add, elementwise, reduction

목표: RMSNorm/FlashAttention에 들어가기 전에 Triton의 execution model을 몸에 익힌다.

### 구현할 미니 커널

1. vector add
2. elementwise multiply
3. residual add
4. row-wise sum
5. row-wise max
6. row-wise softmax

### 배울 개념

- `@triton.jit`
- `tl.program_id`
- block size
- pointer arithmetic
- mask load/store
- `tl.load`, `tl.store`
- `tl.sum`, `tl.max`
- `tl.exp`
- contiguous vs non-contiguous tensor
- stride handling

### 산출물

- `kernels/vector_add.py`
- `kernels/softmax.py`
- `tests/test_vector_add.py`
- `tests/test_softmax.py`
- `benchmarks/bench_softmax.py`

### 성공 기준

- PyTorch reference와 forward allclose
- 다양한 shape 통과
- benchmark가 돌아감
- profiler에서 직접 작성한 kernel 이름을 볼 수 있음

---

## Phase 3. RMSNorm forward/backward

목표: 첫 번째 실전 학습용 Triton op를 완성한다.

### RMSNorm 수식

입력 `x`의 shape가 `[B, T, D]`라고 하자. 보통 마지막 차원 `D`에 대해 normalize한다.

```text
r = rsqrt(mean(x^2) + eps)
y = x * r * weight
```

각 row를 `D` 길이 벡터로 보고 처리하면 된다.

### Backward 핵심 수식

`dy`가 upstream gradient일 때:

```text
z = dy * weight
s = sum(z * x)
dx = r * z - (r^3 / D) * x * s
dweight = sum_over_rows(dy * x * r)
```

`dweight`는 hidden dimension별로 batch/time 방향 reduction이 필요하다. 처음에는 `dweight`를 PyTorch로 계산하고, 이후 Triton reduction으로 옮겨도 된다. 하지만 포트폴리오 기준으로는 `dweight`까지 Triton으로 작성하는 편이 더 좋다.

### 구현 전략

#### 1단계: forward만 구현

- row 하나를 한 Triton program이 처리
- `BLOCK_SIZE = next_power_of_2(D)`
- `mask = offsets < D`
- accumulation은 fp32
- output은 입력 dtype에 맞춰 저장

#### 2단계: backward `dx` 구현

- forward 때 저장한 `r` 또는 `inv_rms`를 활용
- 저장 메모리를 줄이고 싶으면 backward에서 다시 계산
- 처음에는 안정성을 위해 `r` 저장 권장

#### 3단계: backward `dweight` 구현

- 여러 row가 같은 weight index에 더하므로 reduction이 필요
- 쉬운 방법: partial sums buffer를 만들고 두 번째 kernel에서 reduce
- 고급 방법: atomic add 사용 후 성능 비교

### 테스트

- shape: `[2, 4, 32]`, `[4, 16, 128]`, `[8, 128, 256]`
- dtype: fp32, fp16, 가능하면 bf16
- non-contiguous 입력은 core path에서는 제외하고, 나중에 stride-aware 버전 추가

### 산출물

- `kernels/rmsnorm.py`
- `autograd/rmsnorm_fn.py`
- `modules/norm.py`
- `tests/test_rmsnorm.py`
- `benchmarks/bench_rmsnorm.py`

### 성공 기준

- forward allclose 통과
- backward gradient 비교 통과
- baseline 모델의 LayerNorm/RMSNorm을 Triton RMSNorm으로 교체 후 training loss가 정상 하락

---

## Phase 4. Fused SwiGLU forward/backward

목표: elementwise fusion의 효과를 보여준다.

### SwiGLU 수식

MLP에서 보통 다음 구조를 쓴다.

```text
u = x @ W_up
g = x @ W_gate
h = silu(g) * u
out = h @ W_down
```

여기서 Triton으로 먼저 구현할 부분은 `h = silu(g) * u`다. matmul까지 한 번에 묶는 것은 나중 단계에서 한다.

### 왜 좋은 커널인가

PyTorch로 쓰면 대략 다음 중간 텐서가 생긴다.

```python
sigmoid_g = torch.sigmoid(g)
silu_g = g * sigmoid_g
h = silu_g * u
```

Triton에서는 `g`와 `u`를 한 번 읽고, `h`만 저장하도록 fusion할 수 있다.

### Backward 수식

```text
dh = upstream gradient
h = silu(g) * u

du = dh * silu(g)
dg = dh * u * silu'(g)

silu'(g) = sigmoid(g) + g * sigmoid(g) * (1 - sigmoid(g))
```

### 구현 전략

- forward: elementwise fused kernel
- backward: `du`, `dg`를 동시에 계산하는 fused kernel
- matmul backward는 이 단계에서는 PyTorch 또는 Phase 5 Linear backward에 맡긴다.

### 산출물

- `kernels/swiglu.py`
- `autograd/swiglu_fn.py`
- `modules/mlp.py`
- `tests/test_swiglu.py`
- `benchmarks/bench_swiglu.py`

### 성공 기준

- forward/backward gradient 일치
- PyTorch `F.silu(g) * u` 대비 kernel launch 수 감소
- MLP에 삽입 후 training loss 정상 하락

---

## Phase 5. MatMul / Linear 커널

목표: Tensor Core를 쓰는 compute-bound 연산을 이해한다.

이 단계는 매우 중요하지만 동시에 함정이 있다. PyTorch의 matmul은 cuBLAS/cuBLASLt를 타기 때문에 아주 강하다. 직접 쓴 Triton matmul이 항상 빠르지는 않다. 이 단계의 목표는 “cuBLAS를 이긴다”가 아니라 “blocked GEMM과 backward를 이해하고, 언제 Triton이 유리한지 설명한다”다.

### Linear 수식

```text
Y = X @ W + b
```

training backward:

```text
dX = dY @ W^T
dW = X^T @ dY
db = sum(dY)
```

### 구현 순서

#### 1단계: forward matmul

- 입력: `A[M, K]`, `B[K, N]`
- 출력: `C[M, N]`
- block: `BLOCK_M`, `BLOCK_N`, `BLOCK_K`
- accumulation: fp32
- output: fp16/bf16/fp32
- `tl.dot` 사용

#### 2단계: Linear wrapper

- `[B, T, D]`를 `[B*T, D]`로 view
- `D × out_dim` weight와 matmul
- bias는 별도 fused add 또는 PyTorch로 시작

#### 3단계: backward

- `dX = dY @ W.T`
- `dW = X.T @ dY`
- `db = reduce(dY)`

#### 4단계: benchmark

- PyTorch matmul
- Triton naive blocked matmul
- Triton autotuned matmul
- shape별 비교

### 테스트 shape

| 용도 | M | K | N |
|---|---:|---:|---:|
| tiny debug | 32 | 64 | 64 |
| attention qkv | B*T | D | 3D |
| MLP up/gate | B*T | D | FFN |
| MLP down | B*T | FFN | D |

### 산출물

- `kernels/matmul.py`
- `kernels/linear.py`
- `autograd/linear_fn.py`
- `modules/linear.py`
- `tests/test_matmul.py`
- `tests/test_linear.py`
- `benchmarks/bench_matmul.py`

### 성공 기준

- matmul forward allclose
- linear backward gradient 비교 통과
- 최소 3개 shape에서 PyTorch 대비 latency table 작성
- 빠르지 않더라도 profiler로 이유 설명

---

## Phase 6. QKV projection과 MLP fusion

목표: Transformer block 안의 실제 구조에 맞춰 kernel을 통합한다.

### QKV projection

일반적인 attention에서는 다음을 계산한다.

```text
q = x @ Wq
k = x @ Wk
v = x @ Wv
```

실무에서는 보통 `Wqkv = [Wq, Wk, Wv]`로 합쳐서 한 번의 matmul로 계산한다.

```text
qkv = x @ Wqkv
q, k, v = split(qkv)
```

Triton 프로젝트에서는 두 방향을 모두 실험해볼 수 있다.

1. PyTorch reference: 세 개 Linear 또는 하나의 packed Linear
2. Triton: 직접 구현한 packed matmul
3. 추가 kernel: qkv output layout을 attention-friendly format으로 변환

### Attention-friendly layout

권장 layout:

```text
q, k, v: [B, H, T, Dh]
```

하지만 matmul output은 보통:

```text
qkv: [B, T, 3 * D]
```

따라서 split + reshape + transpose 과정이 필요하다. 이 구간도 은근히 memory-bound 병목이 될 수 있으므로 Triton layout transform 커널을 작성해볼 만하다.

### MLP fusion

SwiGLU MLP는 다음과 같이 구성한다.

```text
up_gate = x @ W_up_gate       # [B*T, 2*FFN]
up, gate = split(up_gate)
h = silu(gate) * up           # Triton fused SwiGLU
y = h @ W_down
```

Core path에서는 activation fusion부터 안정화한다. 이후에는 `up_gate projection + activation`을 더 강하게 fuse할 수 있다.

### 산출물

- `kernels/qkv.py`
- `modules/attention.py`
- `modules/mlp.py`
- QKV layout benchmark
- MLP fusion benchmark

### 성공 기준

- QKV output이 reference와 일치
- split/reshape/transpose 비용을 profiler에서 확인
- Triton layout transform 적용 전후 비교

---

## Phase 7. Causal Attention → Simplified FlashAttention

목표: 프로젝트의 핵심 보스. Attention score matrix를 HBM에 크게 만들지 않는 attention을 직접 구현한다.

### 7.1 먼저 naive attention을 고정한다

PyTorch reference:

```python
scores = q @ k.transpose(-2, -1) / sqrt(head_dim)
scores = scores.masked_fill(causal_mask, -inf)
probs = softmax(scores)
out = probs @ v
```

문제:

- `scores` shape가 `[B, H, T, T]`
- sequence length가 커지면 memory 사용량이 급격히 증가
- softmax 중간 결과도 저장됨

### 7.2 FlashAttention forward 핵심 아이디어

Q, K, V를 block 단위로 읽는다.

```text
for each Q block:
  acc = 0
  m = -inf
  l = 0

  for each K/V block:
    scores = Q_block @ K_block^T
    scores += causal_mask

    m_new = max(m, rowmax(scores))
    p = exp(scores - m_new)
    l_new = exp(m - m_new) * l + rowsum(p)
    acc_new = exp(m - m_new) * acc + p @ V_block

    m = m_new
    l = l_new
    acc = acc_new

  O_block = acc / l
```

이 방식은 softmax를 streaming/online 방식으로 계산하기 때문에 전체 `[T, T]` score matrix를 HBM에 저장하지 않는다.

### 7.3 초기 제약

처음 구현은 제약을 강하게 둔다.

- causal attention만 지원
- dropout 없음
- head_dim은 32 또는 64부터
- sequence length는 128/256부터
- fp16 input, fp32 accumulation
- mask는 causal mask만
- variable length batch 없음
- RoPE 없음

제약을 줄이는 것은 나중 일이다. 처음부터 일반화하면 완성하기 어렵다.

### 7.4 Backward 계획

FlashAttention backward는 난도가 높다. 그래서 단계적으로 간다.

#### 단계 A: forward-only FlashAttention

- inference/generation에서만 사용
- training에서는 PyTorch attention 유지
- forward correctness와 memory 절감 확인

#### 단계 B: attention backward 수식 이해

naive attention 기준:

```text
O = P V
P = softmax(S)
S = Q K^T / sqrt(Dh)

dV = P^T dO
dP = dO V^T
dS = P * (dP - rowsum(dP * P))
dQ = dS K / sqrt(Dh)
dK = dS^T Q / sqrt(Dh)
```

FlashAttention backward에서는 `P`를 저장하지 않기 때문에 forward에서 저장한 `m`, `l`, `O` 등을 이용해 block별로 재계산한다.

#### 단계 C: simplified backward 구현

- `dV`, `dK`, `dQ`를 block 단위로 계산
- 처음에는 성능보다 correctness 우선
- atomic 또는 block reduction 방식 선택
- 작은 shape에서 PyTorch reference와 비교

### 산출물

- `kernels/flash_attn.py`
- `autograd/flash_attn_fn.py`
- `tests/test_flash_attn.py`
- `benchmarks/bench_attention.py`
- attention memory comparison plot

### 성공 기준

- FlashAttention forward output이 naive attention과 일치
- 가능하면 backward gradient도 일치
- `T=256/512`에서 memory 사용량 감소가 profiler에 보임
- end-to-end 모델에 attention op로 삽입 가능

---

## Phase 8. 모델 조립: kernel mode switch

목표: 커널을 하나씩 켜고 끄며 비교할 수 있는 모델을 만든다.

### 추천 플래그

```bash
--kernel_mode ref
--kernel_mode rmsnorm
--kernel_mode rmsnorm_swiglu
--kernel_mode triton_linear
--kernel_mode flash_fwd
--kernel_mode all
```

또는 config에서:

```yaml
kernels:
  rmsnorm: triton
  swiglu: triton
  linear: torch
  attention: torch
  qkv_layout: triton
```

이렇게 해야 어떤 커널이 문제를 일으키는지 빠르게 분리할 수 있다.

### 모델 동등성 테스트

동일한 seed와 동일한 weight로 다음을 비교한다.

- logits
- loss
- parameter gradients
- 한 training step 후 parameter delta

테스트 예시:

```text
ref_model + torch ops
vs
triton_model + one custom op
```

한 번에 모든 커널을 켜지 않는다.

### 산출물

- `model_triton.py`
- `tests/test_model_equivalence.py`
- `tests/test_training_smoke.py`

### 성공 기준

- 커널별 switch 가능
- 각 switch 모드에서 forward/backward smoke test 통과
- 최소 `rmsnorm + swiglu` 조합으로 training loss 정상 하락

---

## Phase 9. End-to-end training과 benchmark

목표: 프로젝트를 “보여줄 수 있는 결과”로 만든다.

### 실험 세트

#### 실험 A: PyTorch baseline

```text
model: Tier 1
kernel_mode: ref
dataset: Tiny Shakespeare
steps: 5k~20k
```

#### 실험 B: RMSNorm Triton

```text
kernel_mode: rmsnorm
비교: loss curve, step time, norm kernel latency
```

#### 실험 C: RMSNorm + SwiGLU Triton

```text
kernel_mode: rmsnorm_swiglu
비교: MLP elementwise launch count, memory traffic
```

#### 실험 D: Attention benchmark

```text
naive attention vs simplified FlashAttention
T: 128, 256, 512, 1024
head_dim: 64
causal: true
```

#### 실험 E: 전체 조립 모델

```text
kernel_mode: all_available
비교: end-to-end tokens/sec, peak memory, trace
```

### 측정 지표

| 분류 | 지표 |
|---|---|
| 정확도 | loss curve, validation loss, generation sample |
| 커널 성능 | latency, bandwidth estimate, TFLOPs estimate |
| 학습 성능 | step time, tokens/sec |
| 메모리 | peak allocated, attention score allocation 여부 |
| profiler | top CUDA ops, kernel launch count, trace screenshot |
| 안정성 | NaN 발생 여부, gradient norm |

### 산출물

- `reports/final_report.md`
- benchmark CSV
- trace JSON
- profiler screenshots
- loss curve plot
- memory comparison plot

---

## 10. 커널 백로그

| 우선순위 | 커널 | 난도 | 목적 | Core 여부 |
|---:|---|---:|---|---|
| 1 | vector add | 1 | Triton 기본기 | 예 |
| 2 | row-wise softmax | 2 | reduction/mask 연습 | 예 |
| 3 | RMSNorm forward | 3 | 첫 실전 op | 예 |
| 4 | RMSNorm backward | 4 | custom autograd 핵심 | 예 |
| 5 | Fused SwiGLU forward | 3 | elementwise fusion | 예 |
| 6 | Fused SwiGLU backward | 4 | 학습 가능하게 만들기 | 예 |
| 7 | MatMul forward | 5 | Tensor Core/tile 이해 | 예/준예 |
| 8 | Linear backward | 6 | full training op | 선택 |
| 9 | QKV layout transform | 4 | attention 준비 | 선택 |
| 10 | Causal FlashAttention forward | 7 | 메모리 최적화 핵심 | 예 |
| 11 | FlashAttention backward | 9 | 최종 보스 | Full |
| 12 | AdamW update | 5 | optimizer fusion | Full |
| 13 | Cross entropy | 5 | LM head memory 절약 | Full |
| 14 | RoPE | 3 | modern LLM 느낌 | Full |

---

## 11. 테스트 전략

### 10.1 커널 단위 테스트

각 테스트는 최소 네 종류를 포함한다.

```text
test_forward_matches_reference
test_backward_matches_reference
test_different_shapes
test_dtype
```

예시:

```bash
pytest tests/test_rmsnorm.py -q
pytest tests/test_swiglu.py -q
pytest tests/test_flash_attn.py -q
```

### 10.2 gradient 비교

`torch.autograd.grad`로 reference gradient를 구하고, custom op gradient와 비교한다.

비교 대상:

- `dx`
- `dw`
- `db`
- `dQ`, `dK`, `dV`
- `dWqkv`

### 10.3 작은 shape brute force

큰 shape에서 allclose가 실패하면 디버깅이 어렵다. 반드시 아래처럼 작은 shape를 만든다.

```text
B = 1
T = 4
D = 8
H = 2
Dh = 4
```

이 크기에서는 intermediate tensor를 직접 print해서 비교할 수 있다.

### 10.4 모델 smoke test

아주 작은 모델로 10~50 step만 학습한다.

성공 기준:

- loss가 NaN이 아니다.
- gradient가 None이 아니다.
- parameter가 update된다.
- reference와 Triton loss가 크게 벌어지지 않는다.

---

## 12. Benchmark 전략

### 11.1 Microbenchmark 원칙

- warmup을 충분히 둔다.
- timing 전후로 `torch.cuda.synchronize()`를 호출한다.
- 같은 input shape/dtype으로 비교한다.
- median, p20, p80 정도를 저장한다.
- 단일 실행 시간 하나만 보고 결론 내리지 않는다.

### 11.2 Benchmark table 예시

| op | shape | dtype | PyTorch ms | Triton ms | speedup | max error |
|---|---|---|---:|---:|---:|---:|
| RMSNorm fwd | 8192×256 | fp16 | 0.00 | 0.00 | 0.00× | 0.00e+00 |
| SwiGLU fwd | 8192×1024 | fp16 | 0.00 | 0.00 | 0.00× | 0.00e+00 |
| MatMul fwd | 4096×256×768 | fp16 | 0.00 | 0.00 | 0.00× | 0.00e+00 |
| FlashAttn fwd | B=4,H=4,T=512,Dh=64 | fp16 | 0.00 | 0.00 | 0.00× | 0.00e+00 |

### 11.3 End-to-end benchmark table 예시

| model | kernel mode | seq len | batch | tokens/sec | step ms | peak memory | val loss |
|---|---|---:|---:|---:|---:|---:|---:|
| Tier 1 | ref | 256 | 32 | 0 | 0 | 0 | 0 |
| Tier 1 | rmsnorm | 256 | 32 | 0 | 0 | 0 | 0 |
| Tier 1 | rmsnorm_swiglu | 256 | 32 | 0 | 0 | 0 | 0 |
| Tier 1 | all_available | 256 | 32 | 0 | 0 | 0 | 0 |

---

## 13. Profiling 리포트 구성

최종 리포트는 코드보다 더 중요하다. 아래 구조를 추천한다.

```text
reports/final_report.md

1. Project Overview
2. Model and Dataset
3. Kernel Implementations
   3.1 RMSNorm
   3.2 SwiGLU
   3.3 MatMul/Linear
   3.4 FlashAttention
4. Correctness Validation
5. Microbenchmarks
6. End-to-End Training Results
7. Profiling Analysis
8. What Worked
9. What Did Not Work
10. Future Work
```

### 반드시 넣을 그림

- baseline loss curve vs Triton loss curve
- RMSNorm latency comparison
- SwiGLU launch count comparison
- naive attention vs FlashAttention memory usage
- PyTorch Profiler trace screenshot
- 가능하면 Nsight Systems timeline screenshot

### 좋은 분석 문장 예시

- “RMSNorm은 hidden dimension이 작을 때 launch overhead 때문에 PyTorch와 차이가 작았지만, larger row count에서는 memory traffic 감소로 Triton fused kernel이 유리했다.”
- “Custom GEMM은 cuBLAS보다 느렸지만, tile/block scheduling과 Tensor Core utilization을 이해하기 위한 실험으로 가치가 있었다.”
- “FlashAttention forward는 `[T, T]` attention score allocation을 제거하여 sequence length 증가에 따른 memory growth를 줄였다.”

---

## 14. 주차별 실행 계획

## Week 1. Baseline과 실험 인프라

목표: 정답지를 만든다.

- [ ] repo 생성
- [ ] 환경 세팅
- [ ] Tiny Shakespeare prepare script
- [ ] PyTorch GPT baseline 구현
- [ ] training loop 구현
- [ ] generation 구현
- [ ] benchmark helper 구현
- [ ] profiler helper 구현
- [ ] baseline loss curve 저장

완료 조건:

- `python -m nanotriton.train --config configs/tiny_shakespeare_ref.yaml` 실행 가능
- 1k~5k step 학습 가능
- checkpoint에서 문장 생성 가능

## Week 2. Triton 기본기와 RMSNorm forward

- [ ] vector add kernel
- [ ] row-wise reduction kernel
- [ ] fused softmax mini kernel
- [ ] RMSNorm PyTorch reference
- [ ] RMSNorm Triton forward
- [ ] RMSNorm forward test
- [ ] RMSNorm microbenchmark

완료 조건:

- RMSNorm forward가 PyTorch와 일치
- benchmark table 생성

## Week 3. RMSNorm backward와 모델 삽입

- [ ] RMSNorm backward 수식 정리
- [ ] `dx` Triton kernel
- [ ] `dweight` Triton kernel 또는 partial reduction
- [ ] `torch.autograd.Function` wrapper
- [ ] model module로 교체
- [ ] training smoke test

완료 조건:

- RMSNorm forward/backward test 통과
- RMSNorm Triton 모델이 loss를 정상적으로 낮춤

## Week 4. Fused SwiGLU

- [ ] SwiGLU PyTorch reference
- [ ] forward fused kernel
- [ ] backward fused kernel
- [ ] custom autograd wrapper
- [ ] MLP module에 삽입
- [ ] launch count 비교

완료 조건:

- SwiGLU forward/backward test 통과
- RMSNorm + SwiGLU 조합으로 training 가능

## Week 5. MatMul / Linear

- [ ] blocked matmul forward
- [ ] autotune 후보 config 작성
- [ ] Linear forward wrapper
- [ ] `dX`, `dW`, `db` backward 구현 또는 일부 PyTorch fallback
- [ ] PyTorch/cuBLAS 대비 benchmark

완료 조건:

- matmul correctness 통과
- 적어도 하나의 Transformer Linear에 삽입 가능
- 성능이 느리더라도 분석 가능

## Week 6. QKV projection과 layout

- [ ] packed QKV projection 실험
- [ ] qkv split/reshape/transpose 비용 측정
- [ ] attention-friendly layout transform kernel
- [ ] attention module에 통합

완료 조건:

- QKV layout transform correctness 통과
- profiler에서 layout overhead 전후 비교 가능

## Week 7. Naive attention과 fused softmax

- [ ] naive attention reference 고정
- [ ] causal mask 처리 확인
- [ ] row-wise causal softmax Triton 구현
- [ ] small attention test
- [ ] attention benchmark harness 작성

완료 조건:

- attention reference가 안정적
- FlashAttention 구현 전 비교 기준 완성

## Week 8~9. Simplified FlashAttention forward

- [ ] online softmax 수식 정리
- [ ] Q/K/V block pointer arithmetic 구현
- [ ] causal mask 구현
- [ ] forward output 비교
- [ ] sequence length별 memory benchmark
- [ ] attention module에 forward-only mode 삽입

완료 조건:

- FlashAttention forward가 naive attention과 일치
- memory usage 감소를 표/그림으로 제시 가능

## Week 10~11. FlashAttention backward 또는 최종 통합

둘 중 하나를 선택한다.

### 선택 A: FlashAttention backward 도전

- [ ] backward 수식 정리
- [ ] dV kernel
- [ ] dQ/dK kernel
- [ ] gradient test
- [ ] training integration

### 선택 B: 포트폴리오 완성 우선

- [ ] RMSNorm/SwiGLU/Linear 중심으로 full training
- [ ] FlashAttention은 forward benchmark로 제시
- [ ] final report 작성
- [ ] README 정리
- [ ] benchmark plot 정리

시간을 충분히 쓰는 독립 프로젝트라면 선택 A까지 가는 것이 최종 목표에 더 잘 맞는다. 다만 FlashAttention backward는 프로젝트의 보스몹이므로, core training path가 안정화된 뒤 별도 milestone로 진행한다.

## Week 12. 최종 리포트와 polish

- [ ] README 작성
- [ ] final report 작성
- [ ] benchmark table 정리
- [ ] profiler screenshot 정리
- [ ] failure analysis 작성
- [ ] future work 작성
- [ ] 발표용 5분 demo script 작성

완료 조건:

- 처음 보는 사람이 repo를 clone해서 baseline과 kernel test를 돌릴 수 있음
- 결과 표와 그래프가 있음
- 프로젝트 목적과 성과가 명확함

---

## 15. 첫 3일 실행 계획

### Day 1

- repo 생성
- environment report script 작성
- Tiny Shakespeare prepare script 작성
- PyTorch GPT skeleton 작성
- config 시스템 작성

### Day 2

- training loop 완성
- validation loss logging
- checkpoint 저장/로드
- generation script 작성
- baseline 1k step 학습

### Day 3

- Triton vector add
- row-wise sum/max mini kernel
- RMSNorm reference 작성
- RMSNorm forward Triton 초안 작성
- `test_rmsnorm_forward_small_shape` 작성

이 3일이 지나면 프로젝트가 “생각”에서 “실행 중인 repo”로 바뀐다.

---

## 16. 주요 리스크와 대응

| 리스크 | 증상 | 대응 |
|---|---|---|
| backward 수식 오류 | loss가 바로 NaN 또는 gradient mismatch | 작은 shape에서 PyTorch gradient와 비교 |
| fp16 numerical error | forward는 맞는데 backward가 불안정 | accumulation fp32, tolerance 조정, eps 확인 |
| pointer arithmetic 오류 | 특정 shape에서만 실패 | block size로 나누어떨어지지 않는 shape 테스트 |
| matmul이 PyTorch보다 느림 | benchmark에서 speedup 없음 | 실패가 아님. tile size, occupancy, L2 reuse 분석 |
| FlashAttention 디버깅 난이도 | output mismatch 원인 불명 | causal=False 작은 버전부터, 그다음 causal=True |
| 전체 모델에서 문제 발생 | 어느 커널 문제인지 모름 | kernel mode switch로 하나씩 켜기 |
| 너무 큰 모델로 시작 | iteration이 느려 디버깅 불가 | Tier 0부터 시작 |

---

## 17. 포트폴리오 README 구조

```markdown
# NanoTriton-LM

## What this is
A tiny GPT-style language model trained with custom Triton kernels.

## Why this matters
This project demonstrates GPU kernel engineering for Transformer training.

## Implemented kernels
- RMSNorm forward/backward
- Fused SwiGLU forward/backward
- Blocked matmul / Linear
- QKV layout transform
- Simplified FlashAttention forward

## Correctness
Table of forward/backward errors.

## Benchmarks
Microbenchmark and end-to-end training results.

## Profiling
Profiler screenshots and analysis.

## How to run
Commands.

## Lessons learned
What was fast, what was not, and why.
```

README에서 가장 중요한 것은 “내가 무엇을 구현했는지”보다 “어떻게 검증했는지”다.

---

## 18. 추천 명령어 설계

```bash
# baseline data
python data/shakespeare_char/prepare.py

# train baseline
python -m nanotriton.train \
  --config configs/tiny_shakespeare_ref.yaml

# run all tests
pytest tests -q

# run one kernel test
pytest tests/test_rmsnorm.py -q

# benchmark one kernel
python benchmarks/bench_rmsnorm.py \
  --dtype fp16 \
  --hidden 256 \
  --rows 8192

# train with Triton RMSNorm
python -m nanotriton.train \
  --config configs/tiny_shakespeare_triton_rmsnorm.yaml

# profile training
python benchmarks/profile_train.py \
  --config configs/tiny_shakespeare_triton_all.yaml \
  --steps 50 \
  --out reports/traces/triton_all.json

# generate sample
python -m nanotriton.generate \
  --ckpt out/tiny_triton/checkpoint.pt \
  --prompt "Once upon a time"
```

---

## 19. 최종 Definition of Done

### Core Done

- [ ] PyTorch GPT baseline 학습 가능
- [ ] RMSNorm forward/backward Triton 구현
- [ ] SwiGLU forward/backward Triton 구현
- [ ] 최소 하나의 matmul/linear Triton 실험
- [ ] simplified FlashAttention forward 구현 또는 최소 benchmark
- [ ] 모든 구현 커널의 correctness test 존재
- [ ] end-to-end training loss curve 존재
- [ ] microbenchmark table 존재
- [ ] profiler screenshot 존재
- [ ] final report 존재

### Full Portfolio Done

- [ ] FlashAttention backward까지 구현
- [ ] Triton model이 baseline과 비슷한 validation loss 달성
- [ ] `T=512` 이상 attention benchmark에서 memory 절감 확인
- [ ] README만 봐도 프로젝트 가치가 바로 이해됨
- [ ] 실패한 최적화까지 정직하게 분석함

---

## 20. 참고 자료

- Triton tutorials: https://triton-lang.org/main/getting-started/tutorials/
- Triton matrix multiplication tutorial: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
- Triton layer normalization tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
- Triton fused attention tutorial: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
- PyTorch extending/autograd notes: https://docs.pytorch.org/docs/stable/notes/extending.html
- PyTorch profiler docs: https://docs.pytorch.org/docs/stable/profiler.html
- nanoGPT repository: https://github.com/karpathy/nanoGPT
- FlashAttention repository: https://github.com/Dao-AILab/flash-attention
- TinyStories paper: https://arxiv.org/abs/2305.07759
- TinyStories dataset card: https://huggingface.co/datasets/roneneldan/TinyStories

---

## 21. 한 문장 결론

이 프로젝트는 “작은 모델 하나를 훈련했다”가 아니라, “Transformer training stack의 핵심 연산을 직접 해부하고, Triton으로 다시 꿰매고, 수치 검증과 profiling으로 증명했다”가 되어야 한다.
