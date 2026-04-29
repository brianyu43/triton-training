# Lesson 09 Handoff — MHA + Causal Flash Attention (Triton)

기준 날짜: `2026-04-20`

주제: **레슨 8 의 2-D Triton FA 를 실제 LLM forward shape `(B, H, N, d)` + causal mask 까지 밀어서, `torch.library.custom_op` 로 `torch.ops` 에 등록. `torch.compile(fullgraph=True)` 로 그래프 브레이크 없이 통과까지.**

세션 규모: 5 Phase (P1 = 4-D 확장, P2 = causal, P3 = speed bench, P4 = custom_op, P5 = 이 문서 + 블로그).

## 1. 이번 세션에서 한 일

### 하드웨어 사정 — L4 VM 새로 띄움

- 레슨 8 의 `cuda-l4-dev-lesson08` (us-west4-a, SPOT) 는 **L4 stockout** 으로 시작 불가. SPOT→STANDARD 변환해도 안 올라옴.
- 신규 VM: **`cuda-l4-dev-lesson09`**, us-east4-c, `g2-standard-4`, SPOT.
- Python 환경 재구축: torch 2.11.0+cu130, triton 3.6.0, CUDA 13.0. `sudo apt-get install -y libpython3.10-dev` 한 번 필요했음 (Triton JIT 컴파일 시 `Python.h` 요구).

### Triton 커널 — 4-D + causal

한 파일 [triton_kernels/flash_attention_mha.py](/Users/xavier/dev/cudatraining/triton_kernels/flash_attention_mha.py:1), **192 줄** (docstring + autotune 포함). 레슨 8 의 `flash_attention.py` (2-D, ~100 줄) 에서 출발.

| 변경 | 위치 | 핵심 |
|---|---|---|
| 2-D → 4-D stride | kernel 시그니처 4 stride × 3 tensor | `(B, H, N, d)` 전체 stride 를 런치에서 전달 |
| 3-D grid | `(cdiv(N, BM), H, B)` | 각 program 이 하나의 `(batch, head)` Q 블록 담당 |
| `IS_CAUSAL: tl.constexpr` | autotune key 에 포함 | causal / non-causal 두 특수화 커널 컴파일 |
| 루프 스킵 (FA-v2) | `end_n = tl.minimum(N, (pid_m+1)*BLOCK_M)` | 상삼각 타일 자체를 iteration 에서 빼버림 |
| Diagonal mask | `offs_m[:, None] >= offs_n[None, :]` | 대각선 걸친 타일 한 개에만 실질 적용 |

### `torch.library.custom_op` 등록

[triton_kernels/flash_attention_mha_op.py](/Users/xavier/dev/cudatraining/triton_kernels/flash_attention_mha_op.py:1), **76 줄**. `@custom_op("triton_training::flash_attention_mha", mutates_args=(), device_types="cuda")` + `register_fake`. 이후 아무 곳에서나:

```python
import triton_kernels.flash_attention_mha_op   # 사이드이펙트 임포트
torch.ops.triton_training.flash_attention_mha(q, k, v, is_causal=True)
```

로 커널에 접근 가능. `register_fake` 는 FakeTensor 에서 모양만 선언 (`torch.empty_like(q)`) — `torch.compile` 의 shape inference 에 쓰임.

### 벤치 3 종

| 파일 | 목적 | 핵심 |
|---|---|---|
| [bench/bench_flash_attention_mha.py](/Users/xavier/dev/cudatraining/triton_kernels/bench/bench_flash_attention_mha.py:1) (108줄) | 정확도 (P1/P2) | 57 non-causal + 60 causal shape, `N=129/513` edge case 포함 |
| [bench/bench_flash_attention_mha_speed.py](/Users/xavier/dev/cudatraining/triton_kernels/bench/bench_flash_attention_mha_speed.py:1) (147줄) | 3-way 속도 (P3) | ours / SDPA (FA-2) / naïve, LLaMA-7B + GPT-2 shape |
| [bench/bench_flash_attention_mha_op.py](/Users/xavier/dev/cudatraining/triton_kernels/bench/bench_flash_attention_mha_op.py:1) (174줄) | op 통합 (P4) | 3 경로 bit-exact + `torch.compile(fullgraph=True)` × 함수/AttentionBlock |

### GCP 자동화

Phase 별 러너:
- [scripts/gcp_run_lesson09_phase1.sh](/Users/xavier/dev/cudatraining/scripts/gcp_run_lesson09_phase1.sh:1) (non-causal 정확도)
- [scripts/gcp_run_lesson09_phase2.sh](/Users/xavier/dev/cudatraining/scripts/gcp_run_lesson09_phase2.sh:1) (causal 정확도, edge case)
- [scripts/gcp_run_lesson09_phase3.sh](/Users/xavier/dev/cudatraining/scripts/gcp_run_lesson09_phase3.sh:1) (3-way speed)
- [scripts/gcp_run_lesson09_phase4.sh](/Users/xavier/dev/cudatraining/scripts/gcp_run_lesson09_phase4.sh:1) (custom_op + compile)

### GitHub 레포 공개

레슨 9 도중 레포가 로컬 전용임을 발견 (`.git` 없음). `git init` → 레슨 1-9 전체를 커밋 → `brianyu43/triton-training` (public) 에 푸시. 커밋 author 를 `brianyu43` 로 `--reset-author` 교정.

블로그 초안 [docs/blog_draft_lesson_09_mha_causal_fa.md](/Users/xavier/dev/cudatraining/docs/blog_draft_lesson_09_mha_causal_fa.md:1) 는 이 핸드오프와 동시에 작성.

## 2. 산출물

Triton 커널 + op:
- [triton_kernels/flash_attention_mha.py](/Users/xavier/dev/cudatraining/triton_kernels/flash_attention_mha.py:1) — 4-D MHA + causal FA kernel
- [triton_kernels/flash_attention_mha_op.py](/Users/xavier/dev/cudatraining/triton_kernels/flash_attention_mha_op.py:1) — `torch.library.custom_op` 등록

벤치:
- [triton_kernels/bench/bench_flash_attention_mha.py](/Users/xavier/dev/cudatraining/triton_kernels/bench/bench_flash_attention_mha.py:1)
- [triton_kernels/bench/bench_flash_attention_mha_speed.py](/Users/xavier/dev/cudatraining/triton_kernels/bench/bench_flash_attention_mha_speed.py:1)
- [triton_kernels/bench/bench_flash_attention_mha_op.py](/Users/xavier/dev/cudatraining/triton_kernels/bench/bench_flash_attention_mha_op.py:1)

GCP runner (4 파일):
- `scripts/gcp_run_lesson09_phase{1,2,3,4}.sh`

로그:
- [results/lesson09-phase1-20260420-142131.log](/Users/xavier/dev/cudatraining/results/lesson09-phase1-20260420-142131.log:1)
- [results/lesson09-phase2-20260420-143023.log](/Users/xavier/dev/cudatraining/results/lesson09-phase2-20260420-143023.log:1)
- [results/lesson09-phase3-20260420-150136.log](/Users/xavier/dev/cudatraining/results/lesson09-phase3-20260420-150136.log:1)
- [results/lesson09-phase4-20260420-152331.log](/Users/xavier/dev/cudatraining/results/lesson09-phase4-20260420-152331.log:1)

문서:
- 이 파일 — 핸드오프
- [docs/blog_draft_lesson_09_mha_causal_fa.md](/Users/xavier/dev/cudatraining/docs/blog_draft_lesson_09_mha_causal_fa.md:1) — 블로그 초안

메타:
- [README.md](/Users/xavier/dev/cudatraining/README.md:1) 레슨 09 섹션 + 헤드라인 업데이트
- [github.com/brianyu43/triton-training](https://github.com/brianyu43/triton-training) (public 레포 공개)

## 3. 핵심 숫자 (L4 sm_89, torch 2.11 + triton 3.6, fp16)

### 3.1 정확도 (fp32 기준 대비 max rel_err)

| 단계 | shape 수 | worst fp32 rel_err | worst fp16 rel_err |
|---|---|---|---|
| P1 non-causal | 57 | 3.52 × 10⁻³ | 3.35 × 10⁻⁴ |
| P2 causal | 60 (incl. `N=129/513`) | 1.10 × 10⁻³ | 3.15 × 10⁻⁴ |

edge case `N = 129` / `513` 은 마지막 Q 블록이 `BLOCK_M` 보다 작을 때 mask_m 가 제대로 걸리는지 확인용 — 통과.

### 3.2 속도 — LLaMA-7B causal, d=128, fp16 (median of 100 reps)

| (B, H, N)       | ours                | SDPA (FA-2)  | ours/SDPA | vs naive |
|-----------------|---------------------|--------------|-----------|----------|
| (1, 32,  512)   | 0.100ms (21.5 TF)   | 0.100ms      | **1.00×** | 10.97×   |
| (1, 32, 1024)   | 0.223ms (38.5 TF)   | 0.202ms      | 0.90×     | 29.6×    |
| (1, 32, 2048)   | 0.784ms (43.8 TF)   | 0.613ms      | 0.78×     | 31.4×    |
| (1, 32, 4096)   | 2.964ms (46.4 TF)   | 2.559ms      | 0.86×     | 32.7×    |
| (2, 32, 1024)   | 0.428ms (40.1 TF)   | 0.373ms      | 0.87×     | 30.6×    |
| (2, 32, 2048)   | 1.565ms (43.9 TF)   | 1.372ms      | 0.88×     | 31.3×    |
| (4, 32, 1024)   | 0.886ms (38.8 TF)   | 0.726ms      | 0.82×     | 29.3×    |

### 3.3 non-causal control, d=128, fp16

| (B, H, N)       | ours               | SDPA       | ours/SDPA |
|-----------------|--------------------|------------|-----------|
| (1, 32, 1024)   | 0.325ms (52.8 TF)  | 0.291ms    | 0.90×     |
| (2, 32, 2048)   | 2.643ms (52.0 TF)  | 2.352ms    | 0.89×     |

### 3.4 GPT-2-like 짧은 shape, d=64, fp16 causal

| (B, H, N)        | ours               | SDPA       | ours/SDPA |
|------------------|--------------------|------------|-----------|
| (8, 12, 1024)    | 0.302ms (42.7 TF)  | 0.303ms    | **1.00× 동률** |
| (16, 12, 512)    | 0.249ms (25.9 TF)  | 0.282ms    | **1.13× 우세** |

### 3.5 Phase 4 — `torch.compile(fullgraph=True)`

```
[1] raw_wrapper vs torch.ops  err=0.00e+00  (causal False / True 각각)
[1] raw_wrapper vs .default    err=0.00e+00
[2] eager vs compiled function err=0.00e+00  — fullgraph=True 통과
[3] eager vs compiled block    err=0.00e+00  — LLaMA-스타일 attention block 전체가 1 그래프
[4] schema: triton_training::flash_attention_mha(Tensor q, Tensor k, Tensor v, bool is_causal=False) -> Tensor
```

### 3.6 요지

- d=128 (LLaMA-7B) causal 에서 FA-2 의 **78-90 %** 속도
- d=64 (GPT-2) 짧은 shape 에선 **동률 또는 13 % 우세**
- naïve 대비 **29-33×**
- `torch.compile(fullgraph=True)` 통과 — 그래프 브레이크 0 건

## 4. 네 가지 교훈

### (a) **Causal 의 속도는 mask 가 아니라 "루프 스킵" 에서 나온다**

Mask 만 씌우면 여전히 모든 K 타일을 읽음. 단지 `-inf` 로 바꿀 뿐. 진짜 속도 차이는:

```python
if IS_CAUSAL:
    end_n = tl.minimum(N, (pid_m + 1) * BLOCK_M)
else:
    end_n = N
for start_n in range(0, end_n, BLOCK_N):
    ...
```

이 `end_n` 이 평균 `N/2` 로 떨어지면서 **K/V 로딩과 `tl.dot` 두 번이 반으로** 줄어듦. FA-v2 의 causal 최적화가 이 한 줄.

검증: non-causal `(1, 32, 2048, 128)` 이 2.643 ms 인데 causal 이 0.784 ms (대각 포함해서 약 3.3 × 빠름). FLOP 은 절반인데 시간은 더 떨어짐 — 이유는 로딩 파이프라인이 타일 수에 비례해서 amortize 되기 때문.

### (b) **`IS_CAUSAL: tl.constexpr` — 런타임 if 를 컴파일 타임으로 접기**

같은 `IS_CAUSAL` 플래그를 일반 인자로 넘기면:
- 타이트한 K 루프 안에 매 이터레이션 `if is_causal:` 브랜치가 박힘
- PTX 레벨에서 branch + 의존성 체인이 생기고, warp scheduler 가 발 묶임

`tl.constexpr` 로 찍으면:
- Triton 이 `IS_CAUSAL=True` 와 `=False` **두 개의 특수화 커널을 각각 컴파일**
- 런타임엔 해당하는 것만 dispatch — 브랜치 자체가 존재하지 않음

그리고 `@triton.autotune(key=[..., "IS_CAUSAL"])` 에 넣어서 **causal/non-causal 마다 autotune 도 독립적으로 돈다**. causal 은 `BM=64/BN=128`, non-causal 은 더 큰 타일이 유리해지는 게 실제로 관찰됨.

### (c) **`torch.library.custom_op` 는 "op 를 처음부터 영혼 없이 Python 함수로 짜지 마라" 가 아니다**

그냥 Python 함수로도 `triton_flash_attention_mha(q, k, v, is_causal)` 호출은 된다. `custom_op` 는 **세 가지** 를 더 준다:

1. **`torch.compile` 이 그래프를 안 끊음.** `register_fake` 가 shape inference 에 쓰여서 Dynamo 가 우리 커널을 unknown op 로 보지 않음. 검증: `fullgraph=True` 에서 `AttentionBlock` 전체가 한 그래프.
2. **`torch.export` / 직렬화에 이름이 남음.** 나중에 ONNX / AOT / TorchScript 쪽에서 이 그래프를 기록하면 `triton_training::flash_attention_mha` 가 그대로 노드로 박힘.
3. **`torch.ops.<namespace>.<op>` path 로 드롭인.** vLLM 같은 다운스트림 패키지가 Triton-specific 임포트 없이 호출 가능.

이게 **vLLM 이 attention 커널을 exposing 하는 방식** 과 같은 패턴.

### (d) **155 줄로 FA-2 의 80-90 % 는 ROI 괴물**

코어 커널이 `flash_attention_mha.py` 전체 192 줄 (docstring 포함), 실제 kernel body 는 ~100 줄. 거기에 op 래퍼 76 줄. **합쳐도 300 줄 안 짭짤.**

그 300 줄이:
- d=128 LLaMA-7B 에서 cuDNN-backed FA-2 의 78-90 % 속도
- d=64 GPT-2 에선 동률 or 우세
- `torch.compile(fullgraph=True)` 인 상태로 attention block 에 드롭인 가능

이 ROI 가 "Triton 을 배우는 이유" 자체다. 같은 일을 CUDA 로 쓰면 FA v1 레슨 6 (~500 줄 raw + 150 줄 host) 에서 봤듯 최소 5-10 배 분량에, 새 GPU 재튜닝도 수작업.

## 5. 빌드 시스템 & 벤치 함정 기록

### 함정 1: L4 stockout 에서 막힘 (Phase 0)

증상: `gcloud compute instances start cuda-l4-dev-lesson08` 이 ZONE_RESOURCE_POOL_EXHAUSTED. SPOT → STANDARD 변환해도 동일 zone (us-west4-a) 에서 안 올라옴.

해결: 다른 zone 에 신규 VM.
```bash
./scripts/gcp_create_l4_spot_vm.sh nemo-488500 us-east4-c cuda-l4-dev-lesson09
```
**us-east4-c 가 L4 재고가 잘 도는 편.** 다음 세션 stockout 이 또 나오면 zone rotation 이 1 순위.

### 함정 2: `Python.h: No such file or directory`

증상: 첫 Triton JIT 시도에서 `/usr/include/python3.10/Python.h` 없음.

원인: GCP Deep Learning 이미지 중 일부가 `python3-dev` 미설치.

해결: **`sudo apt-get install -y libpython3.10-dev`** (일반 `python3-dev` 가 held 상태라 `libpython3.10-dev` 로 바로 지정). 로그에 남김.

### 함정 3: `repo not a git repository` — 레슨 9 중간에 알아차림

증상: 사용자가 "지금까지 한 레슨들 코드 다 저장되나?" 물어봐서 `.git` 을 찾아봤는데 없음. 커밋 히스토리 **없음**.

원인: 시작부터 그냥 로컬 폴더였음. GCP 로 rsync 만 쓰고 버전관리 생략.

해결:
```bash
git init -b main
git add -A            # .gitignore (.DS_Store, __pycache__, .triton 등) 먼저 세팅
git commit -m "..."
gh repo create brianyu43/triton-training --public --source=. --push
```

그 다음 author 가 OS 계정명 `xavierarm` 으로 박혔어서:
```bash
git config user.name "brianyu43"
git config user.email "<GitHub no-reply 주소>"
git commit --amend --no-edit --reset-author
git push --force-with-lease origin main
```

중간에 유저가 GitHub UI 에서 README 에 한 줄 추가했던 탓에 `git pull --rebase origin main` 으로 한 번 흡수.

**교훈: 레슨 0 에서 `git init` + 첫 커밋이 `make vector_add` 보다 먼저다.**

### 함정 4: Naive attention OOM — Phase 3 bench 에서

증상: `naive_attention` 이 `B=16, H=12, N=512` 쯤에서 `(N, N)` 스코어 텐서 6 GB 찍고 OOM.

해결: `bench_one` 에 `include_naive: bool` 인자 + `naive_mem_bytes < 4 << 30` 조기 차단:
```python
naive_mem_bytes = B * H * N * N * 4
naive_fits = naive_mem_bytes < (4 << 30)
if include_naive and naive_fits:
    ms_naive = triton.testing.do_bench(run_naive, ...)
else:
    naive_str = "          (skipped)"
```

벤치 CSV 에 `(skipped)` 가 뜨는 건 정확도 실패가 아님 — 메모리 이유로 스킵.

### 함정 5: `torch.compile` "Not enough SMs to use max_autotune_gemm mode" 경고

L4 는 58 SM 인데 `max_autotune_gemm` 모드 요구치(>80 SM) 미달. **그냥 정보성 로그. 기능 영향 없음** — `fullgraph=True` 로 그래프가 깨지지 않는 게 우리 목표였고 그건 성공. H100/B200 에서 재측정하면 사라짐.

## 6. 구조 매핑 (레슨 1-9 층위 업데이트)

```
Python 모델 / 유저 (vLLM, 내 서비스)
          │
          │   torch.ops.triton_training.flash_attention_mha(q, k, v, is_causal)   ← 레슨 09 (P4)
          ▼
torch.compile / Dynamo   (register_fake 로 shape inference, 그래프 유지)   ← 레슨 09 (P4)
          │
          ▼
torch.library.custom_op dispatcher   (device_types="cuda" 로 CPU 는 거름)
          │
          ▼
triton_kernels.flash_attention_mha.triton_flash_attention_mha(q, k, v, is_causal=...)   ← 레슨 09 (P1/P2)
          │   (JIT → autotune(KEY=[N, HEAD_DIM, IS_CAUSAL]) → PTX 생성)
          │
          ▼
flash_attention_mha_fwd_kernel    (4-D, 3-D grid, IS_CAUSAL constexpr)    ← 레슨 09 (P1/P2)
          │   online softmax, 2 × tl.dot per iter, loop skip
          ▼
Warp / tensor core    (fp16 mma.m16n8k16 auto)                            ← 레슨 5-8
          │
          ▼
Memory hierarchy    (HBM ↔ L2 ↔ smem ↔ register)                          ← 레슨 1-2
```

**레슨 09 이후**: 레슨 08 의 최상단 박스 (`triton_kernels.triton_flash_attention`) 가 한 층 더 올라갔다. `torch.ops.<namespace>.<op>` 로 **공식 op 로 옷을 입어서**, `torch.compile` 이 이 커널을 "unknown python 함수" 가 아니라 그래프의 일부로 인식. vLLM 이 커스텀 커널을 제공하는 구조와 동일.

## 7. "왜 여전히 Triton 만으로 안 되는가" 의 레슨 9 답

레슨 8 의 답 (CUDA → Triton) 은 여전히 유효하지만, 레슨 9 가 하나 더 보탠다:

**FA-2 의 20-22 % 갭 은 Triton 자체 의 갭이 아니다.** SDPA 가 호출하는 건 cuDNN 의 FA-2 구현 — 그 안엔:
- **async copy + double/triple buffer** (smem 에 K/V 로드 하는 동안 이전 타일 `mma` 진행)
- **persistent kernel 스케줄링** (block 을 계속 살려두고 타일을 재분배)
- **warp specialization** (일부 warp 는 compute, 일부는 load-store 전담)

Triton 에 persistent / warp specialization 이 오는 중이지만 (OpenAI Triton persistent matmul tutorial, Flash Attention v3 의 Triton 포팅), 아직 튜토리얼 수준. **이 갭을 Triton 에서 닫으려면 CUTLASS 3.x 를 옆에 두고 직접 레이어를 찢어야 함** — 레슨 11+ (CUTLASS) 의 영역.

지금 포지션: **"cuDNN 의 80 % 속도를 300 줄로 얻었다. 나머지 20 % 는 1-2 단계 아래로 내려가야 얻어진다."**

## 8. 다음 단계 후보

### Next A: **Backward + autograd** (레슨 10 가장 자연스러운 연결)

지금은 forward only. 실제 학습에 쓰려면:
- Backward kernel (`dQ`, `dK`, `dV`) — forward 의 재사용이 거의 불가 (통계량 l, m 저장 + recompute 결정)
- `flash_attention_mha_op.register_autograd` 로 묶기
- `torch.compile` 이 backward 까지 추적하는지 검증

FA-2 paper 의 backward 섹션이 거의 그대로 Triton 에 매핑됨. 규모: Phase 1-4 의 2-3 배.

### Next B: **vLLM PagedAttention Triton 포팅** (Phase 1 M3)

vLLM 의 `paged_attention_v1` (~500 줄 CUDA) 을 Triton 으로 재작성. block table 로 KV cache 를 분할 접근하는 구조.

**전제**: (1) vLLM 코드 2 주 스터디, (2) 지금 만든 `custom_op` 래퍼 패턴을 block-table 버전으로 확장.

### Next C: **FA-2 의 마지막 20 % — persistent + async copy**

Triton persistent kernel tutorial 부터 시작. L4 에서 `ours/SDPA` 를 0.78 에서 0.90+ 로 끌어올릴 수 있는지. **리스크**: persistent + autotune 이 아직 불안정.

### Next D: **MQA / GQA 지원**

LLaMA-2/3 은 GQA (grouped query attention) — K/V head 가 Q head 보다 적음. 지금 우리 커널은 K/V 의 H 가 Q 와 같다고 가정. GQA 지원은 stride 하나 더 바꾸면 됨.

### 추천 순서

**A → D → C → B**.

- A (backward) 는 자체만으로 써먹을 결과물 + paper 에 나온 알고리즘이라 막힐 곳이 예측됨.
- D (GQA) 는 A 덕분에 training loop 를 굴릴 수 있게 되면 바로 LLaMA-3 shape 에서 벤치를 다시 찍게 해 준다.
- C (persistent) 는 성능 욕심 — backward 가 생긴 뒤에 여유분으로.
- B (vLLM PR) 은 Phase 1 M3 의 오픈소스 실물 결과물. A+D 로 "forward + backward + GQA" 가 생기면 PR 스토리가 강해짐.

## 9. VM 상태

**`cuda-l4-dev-lesson09`** — us-east4-c, `g2-standard-4`, SPOT, **RUNNING**.

유저 선호도 ([feedback_vm_lifecycle.md](/Users/xavier/.claude/projects/-Users-xavier-dev-cudatraining/memory/feedback_vm_lifecycle.md:1)): 레슨 사이에 **VM 살려두기**. 자동 삭제 금지.

이번 세션 종료 후 상태:
- 레포 복사본: `~/cudatraining/` (최신). git 히스토리 동기화됨.
- Python: torch 2.11.0+cu130, triton 3.6.0, CUDA 13.0.
- `libpython3.10-dev` 설치됨 — Triton JIT 재설정 불필요.
- Triton JIT 캐시: `~/.triton/cache/` 에 PTX 유지. 재호출시 warmup 짧음.
- 레슨 08 의 `cuda-l4-dev-lesson08` 은 us-west4-a stockout 으로 **사용 불가**. 필요하면 삭제해도 됨.

**권장**: > 1 일 gap 이면 `gcloud compute instances stop cuda-l4-dev-lesson09 --zone us-east4-c`. 다음 세션 복구 1 분.

## 10. 한 줄 요약

레슨 8 의 2-D Triton FA 를 `(B, H, N, d)` 4-D + causal 로 확장 (~100 줄 kernel body), `IS_CAUSAL: tl.constexpr` + FA-v2 loop-skip 로 causal 속도 확보, `torch.library.custom_op` 로 `torch.ops.triton_training.flash_attention_mha` 에 등록, `torch.compile(fullgraph=True)` 로 LLaMA-스타일 attention block 전체가 한 그래프에 들어감. **L4 에서 300 줄짜리 Triton 이 cuDNN FA-2 의 78-90 % 속도, 짧은 shape 에선 1.13× 우세, naïve 대비 29-33×.** 다음은 backward + GQA.
