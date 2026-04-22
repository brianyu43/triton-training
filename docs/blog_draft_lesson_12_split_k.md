# vLLM 의 Paged Attention V2 를 Triton 으로 — MQA 의 SM 점유율 구멍을 split-k 로 닫기

_(Lesson 12 · 2026-04-22, 카프카-서명 서체 느낌)_

지난 편 ([vLLM 의 Paged Attention 을 Triton 으로 다시 짜보고, vLLM 의 실수를 반복했다](blog_draft_lesson_11_paged_attention.md)) 에서 가장 불편했던 결과는 한 줄로 이랬다.

> MQA (B=16, H=32, H_kv=1, ctx=4k) 에서 우리 paged 커널이 SDPA 대비 **+85 %** 느리다 — 이건 구조적 결함이다.

Lesson 12 를 이 한 줄 때문에 열었다.

(주의: 위의 "+85 %" 는 Phase 4 가 끝난 시점에 한 번 찍고 지나간 값이다. 이번 세션 시작할 때 같은 shape 을 **block_size=16 에 고정** 하고 다시 찍으니 **+646 %** 였다. block_size 가 다르면 gap 이 이만큼 달라진다 — 원래 Phase 4 에서는 best block_size 기준이었고, 이번 세션은 auto-dispatch 비교를 위해 bs 를 고정했다. 어느 쪽 숫자든 결론은 같다: MQA 는 structurally 느리다.)

---

## 1. 결함의 정체를 한 줄로

MQA 에서 single-pass grid 는 `(B, H_kv) = (16, 1) = 16 programs`. L4 에는 SM 이 **58 개**. 16 / 58 ≈ 28 % 점유율. 나머지 42 SM 은 놀고 있다.

이 놀고 있는 SM 을 어떻게 깨울 것이냐가 이 세션의 전부다. Block size 를 8~128 로 스윕해도 bump 가 거의 없다. Parameter tuning 으로 닫힐 구멍이 아니다. **아키텍처** 를 바꿔야 한다.

vLLM 의 v1 → v2 전환이 정확히 이 문제에 대한 그들의 답이다.

## 2. vLLM v2 의 설계 — ctx 축을 쪼개서 2 개 커널로

[paged_attention_v2.cu](https://github.com/vllm-project/vllm/blob/main/csrc/attention/paged_attention_v2.cu) 의 핵심은 두 가지:

1. **Forward kernel**: `grid = (num_heads, num_seqs, max_num_partitions)`. 기존 v1 의 2-D grid 에 **ctx 축의 partition 번호** 를 3 차원으로 추가. `PARTITION_SIZE = 512`.
2. **Reduce kernel**: `grid = (num_heads, num_seqs)`. Forward kernel 이 각 partition 마다 기록한 (max, lse, partial_out) 을 online softmax 로 재결합.

이걸 Triton 으로 옮기면, 새 커널 2 개가 필요하다:

- `paged_attention_split_kernel` — grid `(B, H_kv, SEGMENTS)`. 각 프로그램은 자기 segment 의 `PARTITION_SIZE=512` 토큰만 처리. **정규화되지 않은** `(m_i, l_i, acc)` 을 scratch 에 기록.
- `paged_attention_reduce_kernel` — grid `(B, H_q)`. SEGMENTS 축을 다 읽어서 `alpha = exp(m_s - m_global)` 로 재결합. 최종 normalized output 저장.

Lesson 11 에서 짜놓은 single-pass 커널을 **그대로 두고 옆에 추가** 했다. 이유는 두 가지:
- Dense shape (B\*H_kv ≥ SM 의 절반) 에서는 single-pass 가 더 빠름. Reduce kernel 의 launch overhead 를 낼 이유가 없음.
- Auto-dispatch 로 "현재 shape 에 맞는 것을 고르자" 가 원칙.

## 3. Online softmax 의 "segment 간 재결합" 은 이미 알던 식이다

Split-k 재결합 수식은 사실 새로운 게 아니다. Lesson 09 에서 **같은 공식** 을 이미 썼다 — 거기서는 "block 간 재결합" 이었고, 여기서는 "segment 간 재결합" 이라는 것만 다를 뿐이다.

_segment s_ 가 자기 부분의 `m_s`, `l_s`, `acc_s` 를 기록한다. 재결합은:

```
m_global = max(m_s over s)
alpha_s  = exp(m_s - m_global)
l_global = sum(alpha_s * l_s)
acc_global = sum(alpha_s * acc_s)      # 각 segment 의 acc 를 rescale 해서 합산
out = acc_global / l_global
```

Lesson 09 의 block-내부 online softmax 를 정확히 한 레벨 outward 로 올린 셈. 이 공식을 안다는 것의 이점은 reduce 커널이 **10 줄짜리** 로 끝나는 것이다.

## 4. "Invalid segment" 는 수학적으로 "empty segment" 와 같다

Split kernel 의 grid 는 `(B, H_kv, SEGMENTS)` 인데 SEGMENTS 는 **최장 sequence** 기준이다. Sequence 가 짧아서 일부 segment 가 빈 경우?

빈 segment 는:
- Inner loop 가 0 번 돌기 때문에 `m_i`, `l_i`, `acc` 가 초기값 그대로 (`-inf, 0, 0`) 남음.
- 이 값을 그대로 scratch 에 쓴다.
- Reduce 에서 `alpha = exp(-inf - m_global) = 0`, 따라서 `alpha * l = 0`, `alpha * acc = 0` — 자동 무시.

별도의 "empty mask" 로직이 필요 없다. **sentinel 값이 무시 동작과 수학적으로 동등** 한 자리는 코드를 깔끔하게 만든다. 이건 lesson 12 에서 가장 기분 좋은 순간이었다.

## 5. 함정 — `tl.arange` 는 power of 2 만 허용한다

Reduce kernel 은 `tl.arange(0, SEGMENTS)` 로 SEGMENTS 축을 로드해야 하는데, Triton 의 제약:

```
CompilationError: arange's range must be a power of 2
```

Test case: `ctx=513, partition_size=32` → `SEGMENTS = ceil(513/32) = 17`. Pow2 아님.

_순진한 해결_: scratch 를 next_pow2 로 padding. `segments=17 → 32 할당`. 메모리 낭비 + forward grid 도 같이 키워야 함.

_진짜 해결_: **kernel constexpr 만 pow2 로 분리, scratch 는 actual 크기 그대로**.

```python
# kernel 쪽
offs_s = tl.arange(0, SEGMENTS_P2)     # pow2, 17 → 32
mask_s = offs_s < SEGMENTS              # actual
m_s = tl.load(ptr + offs_s * stride, mask=mask_s, other=-float("inf"))
```

패딩 lane 은 `-inf` 로 로드 → 위에서 말한 대로 재결합에서 자동 0. **같은 sentinel trick 이 두 번 먹힘** (empty segment + pow2 padding).

이 수정으로 **32/32 PASS**. single-pass 도 여전히 32/32 PASS (회귀 없음).

## 6. 첫 휴리스틱은 틀렸다

auto-dispatch 의 첫 안:
```python
use_split_k = (B*H_kv < 0.75 * SM_COUNT) and (segments >= 2)
```

벤치 결과를 보니 이게 너무 관대하다:

```
shape                 SP ms   SK ms   auto 선택  auto 결과
---------------------- ------ ------- --------- ----------
llama70b-B4-ctx2k      0.165   0.196   SK        나쁨 (+19 %)
llama7b-B1-ctx1k       0.142   0.196   SK        나쁨 (+38 %)
mqa-B16-ctx4k          0.331   0.196   SK        좋음 (-41 %)
```

LLaMA-70B (B=4, H_kv=8) 에서는 `B*H_kv = 32` 로 heuristic 에 걸렸는데, 실제로는 GROUP=8 이라서 single-pass 의 inner body 가 이미 KV 로드를 8-query-head 로 amortize 하고 있음. SK 의 추가 parallelism 은 이득이 없고 reduce kernel launch overhead 만 추가됨. LLaMA-7B B=1 ctx=1k 도 segments=2 만 나와서 per-segment 작업량이 너무 작음.

_두 번째 안_:

```python
use_split_k = (B*H_kv < 0.5 * SM_COUNT) and (segments >= 4)
```

- `B*H_kv < 29`: LLaMA-70B B=4 (32) 는 탈락, MQA B=16 (16) 만 살아남음.
- `segments >= 4`: ctx=1k 는 탈락 (2 segments), ctx=4k 는 통과 (8 segments).

**결과**: 10 shape 중 9 에서 auto 가 정답을 고름. 남은 1 개 (llama7b-B1-ctx4k) 는 SK 가 17 % 더 빠른데 auto 가 SP 로 고름 — **놓친 이득보다 regression 안 내는 게 더 중요** 하다는 원칙 쪽으로 기울였다.

## 7. 최종 숫자

MQA B=16 H_kv=1 ctx=4k bs=16 fp16, L4:

| 측정 | before (Lesson 11 끝) | after (Lesson 12 끝) |
|---|---|---|
| SDPA ms | 0.044 | 0.044 |
| **paged ms** | **0.331** (single-pass) | **0.197** (split-k) |
| gap vs SDPA | +645 % | **+344 %** |
| paged 자체 speedup | 1.00× | **1.68×** |

**Paged kernel 만 놓고 보면 1.68× 더 빠름**. 구조적 수정이 먹힘.

SDPA 에 대한 gap 은 여전히 크다. 왜? 이건 split-k 로 닫을 수 없는 영역이기 때문이다.

## 8. 왜 SDPA 를 못 따라잡는가 — L2 locality 이야기

SDPA 0.044 ms, bytes_moved = `2·B·H·d + 2·B·H_kv·ctx·d = 2·16·32·128 + 2·16·1·4096·128 = 16.4 MB`. Effective BW = `16.4e6 / 44e-6 = **761 GB/s**`.

L4 의 DRAM peak = **300 GB/s**. 761 은 물리적으로 불가능.

답: **L2 hit**. K/V 총량이 16 MB 인데 L4 의 L2 는 **48 MB** — 완전히 캐시에 들어감. SDPA 는 이 반복 접근을 L2 에서 처리해서 DRAM bound 가 아니라 L2 bound 처럼 보임.

우리 paged 는:
- `block_table` lookup 이 L2 를 추가로 씀.
- Block 단위 gather 가 L2 의 spatial locality 를 방해.
- 측정된 BW: 165 GB/s (DRAM 에 가까움).

즉 SDPA 는 L2 에서 먹고 paged 는 DRAM 에서 먹는다. Split-k 는 **SM 점유율** 문제를 고쳤지 **L2 residency** 문제를 고치지 않는다. MQA 의 남은 gap 은 L2-aware prefetch / pinned block 배치 같은 더 깊은 최적화의 영역 — 다음 세션 거리로 남겼다.

## 9. 일반화 — "파라미터 스윕이 반응하지 않으면 아키텍처를 의심해라"

Lesson 10 에서 배운 "느린 데는 이유가 있다, 그 이유를 metric 으로 확정해라" 를 한 겹 더 깊이 내려간 셈이다.

Lesson 11 Phase 4 의 관찰: **block_size 8, 16, 32, 64, 128 어느 것을 써도 MQA 는 bs=32 기준 0.203 ms 가 최저** — 즉 parameter 공간을 넘나들어도 구멍이 안 닫힘. 이 시점이 "파라미터 스윕으로 풀 문제가 아니다" 라는 결정적 신호였다.

vLLM 도 v1 → v2 전환 시 같은 벽에 부딪혔을 것이다. 그래서 그들의 답이 partition size tuning 이 아니라 **grid 축 자체를 추가** 하는 것이었다. 우리의 답도 같아야 했다.

## 10. 다음 세션 후보

Lesson 11/12 로 paged attention 의 "정상 범위 (GQA)" 는 SDPA 보다 빠르고, "극단 (MQA)" 는 구조적으로 2/3 까지 따라잡은 상태.

그다음 자연스러운 질문:
1. **L2-aware paged**: MQA 의 L2 residency 를 어떻게 흉내 낼까. Pinned block layout? Per-SM L1 staging?
2. **Unified kernel**: vLLM 의 `kernel_unified_attention_2d` 처럼 prefill + decode 를 한 커널로? 분기 하나로 서빙 코드가 단순해짐.
3. **OSS contribution**: 위 두 개는 실제로 vLLM 에 쓸 만한 PR 거리. Lesson 11/12 의 결과를 그대로 evidence 로 활용 가능.

---

## 부록 — 이번 세션의 재현용 명령

```bash
# GCP L4 spot VM 에서
python3 triton_kernels/bench/bench_paged_attention.py
# → single-pass: 32/32 PASS, split-k: 32/32 PASS

python3 triton_kernels/bench/bench_paged_attention_speed.py --compare-paths
# → 경로 비교 표가 뒤에 출력됨
```

로컬에서:
```bash
bash scripts/gcp_run_lesson12.sh nemo-488500 <zone> <vm-name> all
```

전체 소스: `triton_kernels/paged_attention.py` 의 `paged_attention_split_kernel` + `paged_attention_reduce_kernel` + `triton_paged_attention_decode` wrapper. ~480 줄.
