# Lesson 11 — Paged Attention (vLLM-style)

**Goal (phase 1 북극성)**: vLLM / SGLang 핵심 커널을 직접 재구현해서
"서빙 스택 판단 가능" 의 객관 증거를 만든다.
Lesson 09 의 contiguous MHA 를 **KV cache block table** 구조로 재작성.

**Target HW**: L4 (sm_89), fp16.

**예상 기간**: 3 주 (Phase 0~5).

---

## 왜 Paged Attention 인가

- **vLLM 논문 (SOSP '23)** 의 핵심 기여. 이 커널 없으면 실제 LLM 서빙은
  KV cache fragmentation 때문에 GPU 70% 이상을 못 씀.
- **Block table indirection** 만 추가되고 나머지 (online softmax, tile scan)
  는 Lesson 09 와 동일. 즉 **레슨 09 의 자연스러운 확장**.
- 공개 artifact 가 희귀 — vLLM 쓸 줄 아는 사람은 많아도 커널 소스까지
  읽고 diff 를 낸 블로그는 거의 없음.

---

## 데이터 구조 (vLLM 포맷)

### Contiguous (Lesson 09)
```
K: (B, H, N, d)     # 각 batch 의 토큰이 메모리상 연속
V: (B, H, N, d)
```

### Paged (Lesson 11)
```
K_cache: (num_blocks, block_size, H, d)   # 블록 단위로 쪼개진 pool
V_cache: (num_blocks, block_size, H, d)
block_table: (B, max_blocks_per_seq)      # seq → physical block id
context_lens: (B,)                         # 각 seq 의 유효 토큰 수
```

- `block_size` = 보통 16 (vLLM default). 커질수록 fragmentation ↓ 하지만
  prefetch granularity 손해.
- `block_table[b, logical_block_idx] = physical_block_id` 로 간접참조.
- 물리 블록 풀은 sequence 간 공유 가능 (prefix caching, beam search).

### 메모리 레이아웃 선택
vLLM 의 원본 커널은 K 를 `(num_blocks, H, d/x, block_size, x)` 로 저장
(x=8 for fp16) — LDGSTS vectorization 최적화. 우리는 일단 단순 버전
`(num_blocks, block_size, H, d)` 로 시작하고 Phase 3 에서 최적화 고려.

---

## 커널 변화 (Lesson 09 → 11)

| 부분 | Lesson 09 | Lesson 11 |
|---|---|---|
| Q 로딩 | `(B, H, N, d)` 연속 | 그대로 (Q 는 decode 시 1 토큰만) |
| K/V 로딩 | `K[b, h, start_n:end_n, :]` 바로 | block_table 경유 2-step |
| 그리드 | `(cdiv(N, BM), H, B)` | `(B, H)` (decode 의 경우) |
| 루프 | Q block 당 K 전체 스캔 | **logical block 순회** → physical 변환 → 로드 |
| 마스크 | causal triangular | context_len 경계 마스크 |

---

## Phase 구조

### Phase 0 — 이해 + Python reference (2 일)
**목표**: 데이터 구조 이해, correctness oracle 만들기.

**작업**
- `triton_kernels/paged_attention_ref.py` — PyTorch naive 구현
  (loop 로 block_table 돌면서 gather, 그 뒤 표준 attention)
- 작은 케이스 smoke test: B=1, H=1, block_size=16, context=32 (2 blocks)
- Contiguous KV 를 paged 포맷으로 변환하는 헬퍼 `pack_kv_paged()`
- 검증: `torch.allclose(paged_ref_out, naive_contiguous_out)` pass

**산출물**
- `paged_attention_ref.py`
- `tests/test_paged_phase0.py` (또는 bench 스크립트 내 smoke)
- acceptance: rtol=1e-5 (fp32) paged_ref = contiguous naive

---

### Phase 1 — Triton decode kernel v1 (5-7 일)
**목표**: **decode only** (Q = 1 토큰/seq) paged attention Triton 커널.

**제약 단순화**
- Q shape `(B, H, 1, d)` — 각 sequence 마다 현재 스텝의 단일 쿼리
- MHA only (GQA 는 Phase 2.5 로)
- fp16
- Causal 자동 적용 (decode 는 정의상 모든 context 를 봄 = causal)

**커널 윤곽**
```python
@triton.jit
def paged_attention_decode_kernel(
    Q_ptr,              # (B, H, d) — decode 라 N=1 차원 제거
    K_cache_ptr,        # (num_blocks, block_size, H, d)
    V_cache_ptr,        # same
    block_table_ptr,    # (B, max_blocks)
    context_lens_ptr,   # (B,)
    Out_ptr,            # (B, H, d)
    # strides...
    scale,
    BLOCK_SIZE: tl.constexpr,   # 16/32
    HEAD_DIM: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    ctx_len = tl.load(context_lens_ptr + pid_b)
    num_blocks = cdiv(ctx_len, BLOCK_SIZE)
    
    # Q 로드 (단일 벡터)
    q = tl.load(Q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_d)
    
    # online softmax 상태 (scalar가 아니라 per-query-row 지만 여기선 row=1)
    m_i = -inf
    l_i = 0.0
    acc = zeros(HEAD_DIM)
    
    for logical_blk in range(0, num_blocks):
        # 1) block_table 조회
        phys_blk = tl.load(block_table_ptr + pid_b * max_blocks + logical_blk)
        
        # 2) 해당 물리 블록에서 K, V 로드
        k_blk_ptr = K_cache_ptr + phys_blk * stride_kb + pid_h * stride_kh
        k = tl.load(k_blk_ptr + offs_n[:, None] * stride_kn + offs_d[None, :])
        # (BLOCK_SIZE, HEAD_DIM)
        
        # 3) context_len 경계 마스크
        token_idx = logical_blk * BLOCK_SIZE + offs_n
        mask = token_idx < ctx_len
        
        # 4) QK, softmax update, acc update — Lesson 09 와 동일
        ...
    
    tl.store(Out_ptr + ..., acc / l_i)
```

**Phase 1 acceptance**
- `paged_ref` (Phase 0) 와 수치 일치 (rtol=1e-3, atol=1e-3, fp16)
- 최소 10 shape 통과: B ∈ {1, 4, 16}, H ∈ {1, 8, 32}, d ∈ {64, 128},
  context_len ∈ {16, 256, 2048}

---

### Phase 2 — 일반화 + 엣지케이스 (5-7 일) ✅ 2026-04-21
**목표**: "실제 decode 트래픽" 처럼 동작.

**추가 (완료)**
- Variable context length (배치 내 seq 마다 다른 길이) — case 13 (B=4, ctx=[256,1024,2048,777])
- 마지막 블록 partial fill — 이미 Phase 1 에서 통과 (case 1, 5, 9)
- GQA: `H_kv < H` — kernel 에 `GQA_GROUP_SIZE: tl.constexpr` 추가, `kv_head = pid_h // GQA_GROUP_SIZE` 로 간접참조
- MQA: `H_kv == 1` (group == H) — case 15 통과

**결과 (16 shape × 2 dtype = 32/32 PASS)**
- **LLaMA-3-8B**: B=2, H=32, H_kv=8, d=128 (group=4), fp16 max diff 2.44e-04
- **LLaMA-70B**: B=4, H=64, H_kv=8, d=128 (group=8) + 가변 ctx, fp16 max diff 3.05e-05
- **MQA**: B=2, H=16, H_kv=1, d=64 (group=16), fp16 max diff 1.91e-06
- 커널 수정은 총 4 줄: constexpr 인자 1 추가 + `kv_head` 계산 1 + K/V load base 의 `pid_h` → `kv_head` 2곳

---

### Phase 3 — 성능 + 프로파일 (3-4 일)
**목표**: 언제 paged 가 contiguous 대비 빠르고/느린가를 숫자로 고정.

**벤치**
- **A**: Lesson 09 contiguous (N = ctx_len, B 개 sequence)
- **B**: 우리 paged (same 유효 토큰수)
- **C**: `vllm.paged_attention` (설치되면, optional reference)

**스윕**
- `block_size ∈ {8, 16, 32, 64, 128}` — sweet spot 찾기
- `batch ∈ {1, 8, 32, 128}` — 높은 batch 에서 paged 가 진가
- `context_len ∈ {128, 1024, 4096, 16384}`

**ncu 드릴 (Lesson 10 툴체인 재사용)**
- stall reason: indirection 때문에 long_scoreboard 가 늘었나?
- register pressure: 포인터 연산이 regs 를 더 먹는가?
- L1/L2 hit: block_table load 가 캐시 친화적인가?

**acceptance**
- 벤치 테이블 + 3-줄 결론 (어느 조건에서 paged 가 이득/손해)
- ncu diff: contiguous vs paged 의 stall 분포 비교

---

### Phase 4 — vLLM 소스 읽기 (2-3 일)
**목표**: 우리 구현과 vLLM 실전 커널의 차이를 말로 설명 가능.

**작업**
- `git clone https://github.com/vllm-project/vllm && cd vllm/csrc/attention`
- 타겟 파일: `attention_kernels.cu`, `paged_attention_v1.cu`, `paged_attention_v2.cu`
- 주요 차이점 identify:
  - K 레이아웃 `(num_blocks, H, d/x, block_size, x)` 의 x=8 vectorization
  - partition (v2): long context 에서 block 들을 여러 grid 로 나눔
  - fp8 KV cache 지원
  - ALiBi, rotary 통합
- `docs/lesson_11_vllm_source_notes.md` 에 정리

**acceptance**
- vLLM 핵심 커널 2개 읽고 우리것과 5개 이상 구체적 차이점 기록

---

### Phase 5 — 핸드오프 + 블로그 (2 일)
- `docs/lesson_11_handoff_YYYY-MM-DD.md`
- `docs/blog_draft_lesson_11_paged_attention.md`
- README 업데이트 (lesson 표에 11 행 추가)

---

## 디렉토리 구조

```
triton_kernels/
├── paged_attention.py              ← 메인 Triton 커널 (Phase 1+)
├── paged_attention_ref.py          ← PyTorch reference (Phase 0)
├── paged_attention_op.py           ← torch.library custom_op 등록 (Phase 2+)
└── bench/
    ├── bench_paged_attention.py           ← correctness
    ├── bench_paged_attention_speed.py     ← perf vs contiguous vs vLLM
    └── lesson11_ncu_profile.py            ← Phase 3 프로파일 드라이버

docs/
├── lesson_11_plan.md                      ← 이 문서
├── lesson_11_vllm_source_notes.md         ← Phase 4
├── lesson_11_handoff_YYYY-MM-DD.md        ← Phase 5
└── blog_draft_lesson_11_paged_attention.md ← Phase 5

results/
├── lesson11_phase1/   ← correctness logs
├── lesson11_phase3/   ← bench + ncu reports
└── ...
```

---

## 함정 예상 리스트 (미리 기록)

1. **block_table 이 int32 인지 int64 인지** — vLLM 은 int32. Triton 의 `tl.load` 에
   dtype 매칭 주의.
2. **block_size 가 너무 작으면 Triton 이 BLOCK_N 으로 쓸 수 없음** —
   `tl.dot` 의 최소 M/N 제약 (8 또는 16).
3. **partial 마지막 블록** 을 for-loop 바깥으로 빼지 않으면 마스크가 매 iter
   필요 → 느려짐. 대신 compile-time flag 로 분기.
4. **fp16 exp overflow**: online softmax 의 m 이 fp32 이면 괜찮지만 acc 가
   fp16 이면 터짐. 반드시 acc=fp32.
5. **decode 라 Q=1** 인데 `tl.dot` 은 최소 BLOCK_M=16 을 요구 → Q 를 16 으로
   pad 하거나 `tl.sum(q * k)` 수동 dot 으로 작성.
6. **NVIDIA profiling permission**: Lesson 10 에서 겪은 `NVreg_RestrictProfilingToAdminUsers=1`
   는 여기서도 동일. sudo -E ncu 필요.

---

## 블로그 훅 (미리 쓸 말)

- "LLaMA-7B decode, B=32, ctx=2048 에서 우리 paged 는 contiguous 대비
  X.XX×. 근데 X.XX× 는 **'메모리 절약 만큼'** 과 거의 일치 — paged 의 본질은
  속도가 아니라 **packing density**."
- "block_size 8 vs 64 벤치: 작을수록 block_table load 가 점유, 클수록 마지막
  블록 낭비. sweet spot 은 Y (우리 L4 기준)."
- "vLLM 소스 `paged_attention_v2.cu` 의 **reduce across partitions** 트릭 —
  우리 v1 에서 빠진 long-context 최적화."
