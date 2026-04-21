# nsys 와 ncu 로 내 커널 9 개를 뜯어봤더니 — 숫자 뒤에 숨어 있던 것들

*"빠르다/느리다" 를 넘어서, "왜 그 숫자인가" 에 닿는 도구.*

기준 날짜: 2026-04-20  ·  GPU: NVIDIA L4 (Ada Lovelace, sm_89, 24GB)  ·  CUDA 12.9 · nsys 2025.1.3 · ncu 2025.2

---

레슨 1-9 까지 CUDA / Triton 커널 10 개를 짰다. 각각 벤치를 돌리고 "v4 가 v1 보다 200× 빨랐다" 거나 "ours 가 SDPA 의 78 % 속도를 냈다" 같은 숫자를 남겼다.

그런데 **왜 그 숫자인지 정확히 모른 채** 넘어간 부분이 많았다:
- atomic 이 느리다는 걸 알지만, *얼마나* 느린지, 그 "느림" 이 HW counter 어디에 나타나는지?
- pinned vs pageable 의 속도 차이가 *어느 경로* 에서 나는지?
- Triton 으로 짠 FA 가 SDPA 의 78 % 라는데, 그 잃는 22 % 가 *어디에 있는지*?

이번 글은 기존 커널을 한 줄도 안 바꾸고, **`nsys` (timeline profiler) 와 `ncu` (per-kernel metric profiler) 로 뜯어본 기록** 이다. 새 커널을 더 짜는 것보다 이걸 먼저 하는 게 시간당 이득이 더 컸다.

## 1. 두 도구, 두 시각

| 도구 | 보여주는 것 | 오버헤드 | 대표 질문 |
|---|---|---|---|
| **nsys** (Nsight Systems) | 시간축 이벤트 타임라인 (CUDA API, kernel, memcpy, stream sync) | ~1-5 % | "시간축 어디에서 뭐가 기다리고 있나" |
| **ncu** (Nsight Compute) | 커널 한 번의 내부 HW counter (stall reason, tensor pipe, memory sol) | **10-30 × 실시간** (replay) | "warp 가 매 cycle 에 뭐 하고 있나" |

`nsys` 는 "kernel 과 transfer 의 critical path 균형" 을 본다. `ncu` 는 "커널 내부에서 warp 가 놀고 있는지, 계산하는지, 기다리는지" 를 본다. 둘이 겹치지 않는다.

---

## 2. Phase 1 · `nsys` — pageable 은 D2H 에서 특히 느리다

**질문**: lesson 04 에서 pinned memory 가 pageable 보다 빠르다고 했다. 얼마나, 어느 방향에서?

**실험**: `bin/vector_add --n 16M --iterations 5` 을 두 번 — 한 번은 `--pageable`, 한 번은 `--pinned` — `nsys` 아래서 돌린 뒤 `.nsys-rep` 를 로컬로 가져와서 시각화 + CLI stats 로 숫자 추출.

결과 (nsys `cuda_gpu_mem_time_sum`):

| direction | pageable GB/s | pinned GB/s | speedup |
|-----------|---------------|-------------|---------|
| H2D (134 MB) | 4.77 | 12.35 | 2.59× |
| D2H (67 MB)  | **1.33** | **13.19** | **9.91×** |

**놀란 점**: pageable H2D 가 4.77 GB/s 인데 pageable D2H 가 **1.33 GB/s** — 같은 방향 같은 PCIe 인데 **3.6× 느리다**.

이유는 타임라인을 보면 명확:
- **pageable D2H** 는 `device → pinned bounce buffer → pageable host` 의 2-hop. 마지막 `memcpy-to-pageable-memory` 가 SIMD-친화적이지 않고 page fault / cache eviction 이 섞여서 느림.
- **pageable H2D** 는 reverse 이지만, 드라이버가 pinned staging buffer 에 먼저 복사한 뒤 DMA 로 GPU 까지 가져감. 이 "host memcpy → pinned" 구간이 OS 입장에서 seq-read 에 가까워서 상대적으로 빠름.
- **pinned** 는 두 방향 모두 0-hop (DMA 직결) 이라 대칭 — 12-13 GB/s 로 수렴.

L4 PCIe Gen4 x16 의 effective bandwidth 가 ~26 GB/s 이므로 pinned 는 그 ~50 % 에 도달한다. pageable D2H 는 **5 % 에 머무름** — 구조 때문에 나오는 세금.

**그리고 커널 시간은 움직이지 않는다** (pageable 0.834 ms, pinned 0.836 ms). pinning 은 전송 경로만 건드리지 on-device 실행과 무관 — 이 당연한 사실을 숫자로 확인.

**한 줄 결론**: pageable → pinned 전환은 "전송 시간 2× 빨라짐" 으로 모호하게 기억하지 말고, **"D2H 에서 10× 빨라짐"** 으로 기억하자. 실제로 사용자 latency 에서 가장 큰 변화가 거기서 온다.

---

## 3. Phase 2 · `ncu` — atomic 의 "느림" 은 occupancy 가 아니라 `lg_throttle`

**질문**: lesson 02 reduction v1 (`atomicAdd` per thread) 이 v4 (warp shuffle + 1 atomic per block) 보다 수백 배 느린 건 아는데, **그 "느림" 이 정확히 어떤 HW counter 에 드러나는가**?

**실험**: `bin/reduction --n 4M --version {1,4}` 두 가지를 각각 `ncu --set detailed --launch-skip 20 --launch-count 1 -k "regex:reduce_v{1,4}_"` 로 한 번씩 뜬 뒤, stall 분포 + SOL 지표 비교.

결과:

| metric | v1 (atomic per thread) | v4 (shuffle + block atomic) |
|---|---|---|
| Elapsed cycles | **12,085,435** | **55,229** (218× 적음) |
| DRAM throughput | **0.46 %** | **88.2 %** (192× 많음) |
| L2 hit rate | **88.74 %** (!) | 0.95 % |
| Achieved occupancy | 91.17 % | 91.89 % (**거의 같음**) |
| Dominant stall | `lg_throttle` 31.1 % | `long_scoreboard` 84.6 % |

**세 가지 놀라운 점**:

**(1) Occupancy 는 같다.** v1 과 v4 둘 다 91-92 % 로 거의 동일. 직관적으론 "v1 이 atomic 에 막혀서 warp 가 못 뜨지 않나" 싶은데, 사실은 warp 는 다 뜨는데 뜬 채로 **기다리고 있다**. 그게 occupancy 엔 잡히지 않는다.

**(2) DRAM 은 비어 있다.** v1 의 DRAM 이 0.46 %. 이 커널은 **memory-bound 가 아니다**. 대역폭 탓이 아니라는 뜻.

**(3) 그런데 L2 hit 이 88.74 % — 비정상적으로 높다.** 이게 힌트. 모든 thread 가 같은 4-byte accumulator 를 건드리니까 그 cache line 이 L2 에 못 박혀서 계속 hit. 하지만 hit 이 많다고 빠른 게 아니다 — 모든 SM 이 그 한 line 을 두고 싸우는 **직렬화** 가 일어난다.

이게 `lg_throttle` **31.1 %** 로 나옴. "local/global memory throttle" — LSU (load/store unit) 가 atomic path 에서 back-pressure 를 받고 있다는 신호.

v4 에서는 이게 완전히 사라진다. `lg_throttle` 0 %, dominant stall 이 `long_scoreboard` (정상적인 DRAM load 대기) 로 바뀌고, DRAM 이 88 % 까지 차면서 **memory-bound 의 건강한 모양** 이 된다.

**레슨**: "atomic 이 느리다" 는 말은 이 정도 세부로 기억하자 — **"atomic 은 L2 cache line serialization 을 만들고, 그게 `lg_throttle` 로 counter 에 나오고, 그 사이 DRAM 은 빈다"**. 이 세 문장이 같이 있을 때 비로소 "왜 느린지" 가 설명됐다고 할 수 있다.

**추가 재미있는 점**: occupancy 가 높은 상태로 커널이 느린 모양은 "warps 가 뜨지 못해서" 느린 게 아니라 "warps 가 떴는데 못 움직여서" 느린 패턴. 이 모양은 Phase 3 에서 또 한 번 나온다 — 방식만 다르게.

---

## 4. Phase 3 · `ncu` 로 우리 커널 vs SDPA 의 20 % gap 추적

**질문**: lesson 09 에서 Triton 으로 짠 4-D causal FA 가 `F.scaled_dot_product_attention` 의 **78-90 % 속도** 를 냈다. 그 22 % 가 어디에 있나?

**실험**: `B=1 H=32 N=2048 d=128 causal fp16` (LLaMA-7B mid-range, gap 이 가장 컸던 shape). 두 구현을 각각 한 번씩 `ncu` 로 뜨고 metric 비교.

**먼저 발견한 것 — SDPA backend 가 cuDNN 이 아니었다**. kernel 이름:
```
void flash_fwd_kernel<Flash_fwd_kernel_traits<128, 64, 64, 4, 0, 0, half_t, ...>>(Flash_fwd_params)
```
이건 Tri Dao 의 Flash Attention 2 CUDA 구현 — PyTorch 2.11 이 번들로 가지고 있다가 L4 + fp16 + causal 조합에서 디스패치한 것. **cuDNN 아님**. 즉, 우리는 Triton FA 를 **같은 알고리즘의 숙성된 CUDA 구현** 과 비교하게 된다.

핵심 지표:

| metric | ours (Triton) | SDPA (FA-2 CUDA) | 비율 |
|---|---|---|---|
| Elapsed cycles | 1,565,141 | 827,328 | 1.89× |
| Compute (SM) throughput | 39.3 % | **72.1 %** | 1.84× |
| Tensor pipe utilization | 44.6 % | **78.8 %** | 1.77× |
| DRAM throughput | 10.6 % | 20.3 % | 1.92× |
| **Registers per thread** | **255 (spill 직전)** | **184** | 0.72× |
| **Achieved occupancy** | **8.3 %** | **16.2 %** | 1.95× |
| Grid size / block size | 512 / 128 | 1024 / 128 |  |

Stall 분포:

| stall reason | ours | SDPA |
|---|---|---|
| total samples | 78,144 | 42,886 |
| `wait` (MMA output dep) | **38.6 %** | 19.0 % |
| `selected` (issue 됨) | 21.7 % | 13.6 % |
| `math_pipe_throttle` (tensor sat) | 19.4 % | **41.5 %** |
| `short_scoreboard` (reg dep) | **14.9 %** | 2.2 % |

**20 % gap 이 있는 네 군데**:

**(1) Register pressure → Occupancy 반토막**.
ours 는 `BLOCK_M=128` 을 autotune 이 골라서 register 가 255 (literally max, spill 직전). 그 결과 SM 에 **resident warp 수가 절반**. SDPA 는 `BLOCK_M=64` 타일로 184 reg/thread, 2 배의 warp 를 동시에 살려 둔다. Occupancy 8.3 % vs 16.2 %.

**(2) MMA dependency chain (`wait` 38.6 %)**.
`tl.dot` 직후 output accumulator 를 너무 빨리 consume. `num_stages` 가 부족해서 producer MMA 가 아직 끝나지 않은 상태에서 consumer 가 기다림. SDPA 는 이 `wait` 가 19 % 로 절반.

**(3) Register dependency (`short_scoreboard` 14.9 % — SDPA 는 2.2 %)**.
(1) 의 연쇄 — register file 이 꽉 차서 producer-consumer 가 자주 같은 물리 register 를 참조. 작은 latency 에 걸림.

**(4) SDPA 는 이미 "좋은 bottleneck" 에 도달함**.
`math_pipe_throttle` **41.5 %** — tensor core 가 포화 상태. 이게 `wait` 보다 좋은 신호인 이유: "FLOP 을 더 박아야 빨라지는" 구간에 왔다는 뜻. 이 상태에서 더 빠르려면 알고리즘을 바꾸거나 HW 가 바뀌어야 함. 우리는 거기까지 못 감.

---

## 5. 이 세션의 세 가지 교훈

### (a) **Occupancy 는 throughput 이 아니다**

Phase 2 와 Phase 3 의 공통점: occupancy 만 봤으면 틀린 진단을 내렸을 것.
- Phase 2: v1 과 v4 의 occupancy 는 91-92 % 로 같은데 wall cycle 은 218× 차이.
- Phase 3: ours 의 occupancy 가 8.3 % 로 SDPA 의 16.2 % 대비 **절반** 이지만, 이건 "warp pool 이 비어서" 라기보다 "큰 tile + 높은 register pressure" 의 부작용.

occupancy 는 "몇 warp 가 살 수 있느냐" 의 상한. "그 warp 가 뭘 하고 있느냐" 는 별도로 봐야 하고, 그걸 보려면 DRAM / compute / tensor pipe 의 SOL % 와 stall 분포를 같이 봐야 한다.

### (b) **`ncu` 의 stall 분포 = kernel 의 성격 지문**

각 커널이 어떤 stall reason 이 dominant 인지에 따라 성격이 다르다:

- `long_scoreboard` dominant = DRAM / L2 load 대기 — **memory-bound**. 맞는 처방: access pattern, tiling.
- `math_pipe_throttle` dominant = tensor / FP pipe saturated — **compute-bound (healthy)**. 맞는 처방: "더 빨리 가기 어렵다" — 알고리즘 변경 또는 HW 변경.
- `wait` dominant = MMA output dependency — **파이프라이닝 부족**. 맞는 처방: `num_stages` 올리기, accumulator 사용 패턴 재구성.
- `lg_throttle` dominant = LSU atomic / misaligned — **알고리즘 설계 문제**. 맞는 처방: 알고리즘 재설계 (Phase 2 의 reduction v1 → v4 가 정확히 이거).

이 분포를 보고 나서야 **"무엇을 고쳐야 할지"** 가 명확해진다. ncu 없이 이 판단은 못 한다.

### (c) **큰 tile 이 빠를 거라는 직관은 틀릴 수 있다**

Phase 3 에서 autotune 은 `BLOCK_M=128` 을 골랐지만, 그게 L4 의 이 shape 에서 최적은 아니었다. 이유: register pressure 가 warp pool 을 고갈시킴.

작은 tile 의 장점:
- register 적게 씀 → occupancy 올라감
- K/V 재사용 주기가 짧아서 software pipelining 이 잘 됨

큰 tile 의 장점:
- 각 block 이 한 번 데이터 읽고 더 많은 계산 → arithmetic intensity 높음

어느 쪽이 이기는지는 **측정하지 않으면 모름**. 그리고 autotune 이 wall-time 으로 best 를 고르더라도, "그 best 가 HW 를 최대로 쓰고 있는가" 는 ncu 로 확인해야 한다. 지금 우리 autotune 은 wall-time 은 best 를 골랐지만 HW 를 최대로 쓰진 않고 있다 (39 % compute, 44 % tensor).

---

## 6. 내가 받아온 실용적인 도구 체인

이 세션 이후로 새 커널 작성 시 기본 체크리스트가 바뀌었다:

1. 커널 돌려서 wall time 재기
2. `nsys` 로 timeline 떠서 **kernel vs transfer 의 critical path** 확인
3. `ncu --set detailed` 로 DRAM / Compute / Tensor pipe 의 **SOL %** 확인
4. **Stall reason 분포** 확인 — 어떤 stall 이 dominant 인가?
5. dominant stall 에 따라 처방:
   - `long_scoreboard` → tiling / cache 분석
   - `math_pipe_throttle` → 그 자리가 ceiling. 알고리즘 변경 고민
   - `wait` → `num_stages` 또는 register 압박 재구성
   - `lg_throttle` → atomic / misalignment 재설계

그리고 "speedup 을 자랑하기 전에" 적어도 DRAM % 와 dominant stall 은 기록으로 남겨둔다. 반대로 **"왜 느린지 모르는 상태" 에서 발표하는 자료는 이제 만들지 않는다**.

---

## 7. 마지막으로 — 이 세션이 말하는 것

- lesson 04 의 pinning 효과는 **D2H 10×, H2D 2.6×** 로 비대칭이다. nsys timeline 에서 2-hop memcpy path 를 보면 이해된다.
- lesson 02 의 reduction v1→v4 218× 차이는 occupancy 가 아니라 **atomic serialization (`lg_throttle` 31 %)** 때문이다.
- lesson 09 의 우리 FA 가 SDPA 의 78-90 % 에서 잃는 **22 %** 는 register pressure (255 regs → occupancy 절반) + MMA dependency (`wait` 39 %) 때문이다. SDPA 는 이미 tensor pipe throttle (healthy bottleneck) 지점.

이 세 인사이트는 **프로파일링 툴 없이는 얻을 수 없는 해석** 이다. 그리고 이 해석이 있어야 다음 iteration 에서 뭘 고쳐야 할지 말이 된다.

레슨 10 이 "새 커널 0 개 만든" 세션이지만, **다음 레슨 이후의 모든 커널 튜닝의 출발점을 앞당겼다** 는 의미에서 이번 세션이 제일 남는 장사였다.

---

### 부록 — 이 세션의 재현용 커맨드

```bash
# Phase 1 — nsys timeline diff (pinned vs pageable)
./scripts/gcp_run_lesson10_phase1.sh <PROJECT_ID> us-west1-b cuda-l4-dev-lesson10

# Phase 2 — ncu reduction v1 vs v4
./scripts/gcp_run_lesson10_phase2.sh <PROJECT_ID> us-west1-b cuda-l4-dev-lesson10

# Phase 3 — ncu ours vs SDPA
./scripts/gcp_run_lesson10_phase3.sh <PROJECT_ID> us-west1-b cuda-l4-dev-lesson10

# GUI 열람 (Mac):
open -a 'Nsight Systems'  results/lesson10_phase1/vector_add_pinned.nsys-rep
open -a 'Nsight Compute'  results/lesson10_phase3/fa_sdpa.ncu-rep
```

GCP DL image 에서 `ncu` 는 `sudo -E env PATH=$PATH ncu ...` 로 감싸야 perf counter 접근 가능. 드라이버 재부팅은 필요 없다.

### 재료 링크

- 레슨 10 핸드오프: [`docs/lesson_10_handoff_2026-04-20.md`](/Users/xavier/dev/cudatraining/docs/lesson_10_handoff_2026-04-20.md:1)
- Phase 1 summary: [`results/lesson10_phase1/summary.md`](/Users/xavier/dev/cudatraining/results/lesson10_phase1/summary.md:1)
- Phase 2 summary: [`results/lesson10_phase2/summary.md`](/Users/xavier/dev/cudatraining/results/lesson10_phase2/summary.md:1)
- Phase 3 summary: [`results/lesson10_phase3/summary.md`](/Users/xavier/dev/cudatraining/results/lesson10_phase3/summary.md:1)
- 이전 레슨 08, 09 블로그: [`docs/blog_draft_triton_vs_cuda.md`](/Users/xavier/dev/cudatraining/docs/blog_draft_triton_vs_cuda.md:1), [`docs/blog_draft_lesson_09_mha_causal_fa.md`](/Users/xavier/dev/cudatraining/docs/blog_draft_lesson_09_mha_causal_fa.md:1)
