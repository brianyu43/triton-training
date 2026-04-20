# triton-training

CUDA → Triton 커널을 lesson 단위로 쌓는 개인 훈련 레포. 각 커널은 *측정과 해석이 가능한 형태* 로 구현하고, lesson별 handoff 문서에 의사결정/속도/막힌 포인트를 남긴다.

6개월 Phase 1 목표: vLLM / Triton / Flash Attention 같은 GPU 서빙 스택에서 판단 가능한 엔지니어로 착지.
-> 손코딩열심히 
---

## 현재 범위 (Lesson 01 → 09)

| #  | 커널                          | CUDA | Triton    | 핵심 포인트 |
|----|------------------------------|------|-----------|------------|
| 01 | Vector Add                   | ✅   | ✅ (smoke) | 메모리 대역폭 기준선, launch config |
| 02 | Reduction                    | ✅   | ✅        | warp shuffle, 블록 리덕션, atomic 대안 |
| 03 | Softmax                      | ✅   | ✅        | numerically-stable, row-wise |
| 04 | Pinned vs Pageable           | ✅   | —         | H2D/D2H 전송 오버헤드 |
| 05 | Matmul                       | ✅   | ✅        | tiled → block-tiled → warp-tiled |
| 06 | Matmul 확장                  | ✅   | —         | shared memory padding, double buffer |
| 07 | Flash Attention v1           | ✅   | —         | online softmax, 단일 타일 |
| 08 | Flash Attention (Triton)     | —    | ✅        | autotune, fp16/fp32, N=8192에서 우리 CUDA 대비 **6.14×** |
| 09 | MHA + Causal Flash Attention | —    | ✅        | `(B, H, N, d)` 4-D, `IS_CAUSAL` constexpr, 루프 스킵, `torch.ops` 등록 |

각 lesson handoff 문서: [`docs/lesson_XX_handoff_*.md`](./docs/).
블로그 초안 (Triton vs CUDA): [`docs/blog_draft_triton_vs_cuda.md`](./docs/blog_draft_triton_vs_cuda.md).

---

## 헤드라인 결과 (L4 GPU, sm_89)

- **Triton fp16 matmul**: ~54 TFLOPS (cuBLAS 대비 ~77%)
- **Triton fp16 Flash Attention**, N=8192, d=64: lesson 07의 우리 CUDA FA v1 대비 **6.14× 빠름**
- 원인 분해 — autotune (`BLOCK_M/N`, `num_warps`, `num_stages`) + 온라인 softmax + `tl.dot` HW matrix path

> 이 숫자는 L4 (Ada Lovelace, sm_89) 한 장에서만 유효. 다른 GPU에서는 autotune을 다시 타야 의미 있음.

---

## Lesson 09 — 4-D MHA + Causal Flash Attention

- **인터페이스**: `(B, H, N, d)` — 실제 LLM forward shape
- **3-D 그리드**: `(cdiv(N, BLOCK_M), H, B)` → 각 program이 하나의 `(batch, head)` Q 블록 담당
- **Causal mask**: `IS_CAUSAL: tl.constexpr` → Triton이 causal/non-causal 두 커널로 특수화 컴파일, 런타임 브랜치 비용 0
- **FA-v2 루프 스킵**: causal일 때 상삼각 타일을 아예 건너뜀 — `end_n = min(N, (pid_m+1)*BLOCK_M)` → 평균 ~50% K/V 스캔 단축

진행 상태:
- ✅ Phase 1: 2-D → 4-D 확장, non-causal 정확도 통과 (57 shape)
- ✅ Phase 2: causal mask + 루프 스킵, 정확도 통과 (60 shape, `N=129/513` edge case 포함)
- ✅ Phase 3: 3-way speed bench (ours / SDPA / naive) on LLaMA-7B + GPT-2 형태
- ✅ Phase 4: `torch.library.custom_op` 등록, `torch.compile(fullgraph=True)` 통과
- ✅ Phase 5: [lesson 09 handoff](./docs/lesson_09_handoff_2026-04-20.md) + [블로그 초안](./docs/blog_draft_lesson_09_mha_causal_fa.md)

### Phase 3 headline (L4, fp16, median of 100 reps)

| Shape (LLaMA-7B causal, d=128)   | ours        | SDPA (FA-2) | ours/SDPA | vs naive |
|----------------------------------|-------------|-------------|-----------|----------|
| B=1 H=32 N=1024                  | 0.223ms (38.5 TF) | 0.202ms | 0.90× | 30× |
| B=1 H=32 N=2048                  | 0.784ms (43.8 TF) | 0.613ms | 0.78× | 31× |
| B=1 H=32 N=4096                  | 2.964ms (46.4 TF) | 2.559ms | 0.86× | 33× |
| B=4 H=32 N=1024                  | 0.886ms (38.8 TF) | 0.726ms | 0.82× | 29× |

| Shape (GPT-2 causal, d=64)       | ours              | SDPA    | ours/SDPA |
|----------------------------------|-------------------|---------|-----------|
| B=8  H=12 N=1024                 | 0.302ms (42.7 TF) | 0.303ms | **1.00× 동률** |
| B=16 H=12 N=512                  | 0.249ms (25.9 TF) | 0.282ms | **1.13× 우세** |

요지: ~155줄 Triton이 d=128 LLaMA-7B 형태에서 FlashAttention-2 대비 78–90% 속도. d=64에서는 동률 또는 우세. naïve 대비 29–33×.

---

## 빌드 & 실행

### CUDA (lessons 01–07)
```bash
make vector_add
./bin/vector_add --n 67108864 --block-size 256 --iterations 100
```

다른 커널도 같은 패턴: `make reduction`, `make softmax`, `make matmul`, `make flash_attention`.

### Triton (lessons 05, 07, 08, 09)
```bash
# Lesson 09 · correctness (non-causal + causal)
python3 triton_kernels/bench/bench_flash_attention_mha.py

# Lesson 09 · 3-way speed bench (ours vs SDPA vs naive)
python3 triton_kernels/bench/bench_flash_attention_mha_speed.py
```

### 클라우드 VM (GCP)

로컬 Mac (Apple Silicon)에는 NVIDIA GPU가 없어서, 실행은 GCP L4 spot (lessons 07–09) 또는 T4 spot (lessons 01–06)에서 한다:

```bash
# 1회: VM 생성 (G2 family = L4 1장)
./scripts/gcp_create_l4_spot_vm.sh <PROJECT_ID> <ZONE> <VM_NAME>

# lesson 실행마다: 레포 sync + run
./scripts/gcp_run_lesson09_phase2.sh <PROJECT_ID> <ZONE> <VM_NAME>
```

---

## 출력에서 볼 것 (lesson 01 스타일 기준)

프로그램은 아래를 출력한다.

- GPU 이름과 대략적인 theoretical memory bandwidth
- kernel best/avg time
- effective bandwidth
- H2D / D2H copy time
- CPU reference와의 최대 오차

처음에는 숫자 자체보다 아래 질문이 중요하다.

- block size를 바꾸면 왜 거의 안 변하거나, 어느 지점부터만 변하는가
- effective bandwidth가 theoretical bandwidth 대비 몇 퍼센트인가
- kernel time과 H2D/D2H 복사 시간 중 어디가 더 큰가

---

## 디렉토리 구조

```
src/                CUDA 커널 (.cu) — lessons 01–07
triton_kernels/     Triton 커널 (.py) + bench/ 서브디렉토리
docs/               lesson별 handoff 문서 + 블로그 초안
scripts/            GCP VM create/copy/run 스크립트
extension/          (legacy) PyTorch C++ extension — lesson 01 초기 실험
Makefile            CUDA 커널 빌드 타겟
```

---

## 의도

- **공개 artifact 중심** — commit / bench output / handoff 문서 자체가 포트폴리오.
- **학위가 아니라 측정 결과로** — cuBLAS 대비 %, torch SDPA 대비 × 단위로 줄을 긋는다.
- **독학 기록** — 막혔던 지점, 잘못 판단했던 포인트를 lesson handoff에 그대로 남긴다.

## 라이선스

학습용 개인 저장소.
