# Lesson 10 Handoff — Nsight Systems + Nsight Compute 로 1-9 레슨 뜯어보기

기준 날짜: `2026-04-20`

주제: **레슨 1-9 에서 "빠르다 / 느리다" 만 보고 넘겼던 포인트를 `nsys` (timeline) 와 `ncu` (per-kernel metric) 로 뜯어서 *왜* 그 숫자인지 본다.** 새 커널을 더 짜는 세션이 아니라, 이미 짠 커널을 **프로파일링 툴로 해석하는 세션**.

세션 규모: 4 Phase (P1 = lesson 04 nsys timeline, P2 = lesson 02 ncu stall diff, P3 = lesson 09 ours vs SDPA ncu diff, P4 = 이 문서 + 블로그).

## 1. 이번 세션에서 한 일

### 왜 이 세션이 필요한가

레슨 1-9 까지 나온 숫자들:
- 레슨 04: pinned 가 pageable 보다 "빠르다" — 얼마나? 왜?
- 레슨 02: v4 (warp shuffle) 가 v1 (atomicAdd) 보다 "훨씬 빠르다" — atomic serialization 이 진짜 병목인가, 아니면 occupancy / 다른 이유인가?
- 레슨 09: ours = SDPA 의 78-90 % — 그 **20 % 가 어디 있는지** 말로는 추측 가능하지만, 계량한 적이 없음.

각각의 질문은 "느낌적인 느낌" 으로 답을 아는 상태였지만, **숫자로 박힌 적이 없음**. nsys / ncu 가 이걸 메꾸는 도구.

### 하드웨어 사정 — 또 L4 stockout

- 레슨 9 의 `cuda-l4-dev-lesson09` (us-east4-c, SPOT) 를 재시작하려니 **또** stockout. us-east4-c, us-central1-{a,b,c}, us-west1-a 다 STOCKOUT.
- us-west1-b 에서 빈자리 찾음 → `cuda-l4-dev-lesson10` 신규 생성.
- 이미지가 `common-cu129-ubuntu-2204-nvidia-580` (CUDA 12.9, driver 580). **DL 이미지인데 `make` / `libpython3.10-dev` 가 기본으로 없어서** `sudo apt-get install build-essential libpython3.10-dev` 한 번 필요.
- Phase 3 위해 `torch 2.11.0+cu128` + `triton 3.6.0` 재설치.

### 프로파일링 툴 개념 정리

| 툴 | 시각 | 오버헤드 | 답하는 질문 |
|---|---|---|---|
| **nsys** (Nsight Systems) | timeline (ms~μs 단위 이벤트) | ~1-5 % | "시간축에서 뭐가 언제 실행되고 무엇이 기다리나" — CUDA API, 커널, memcpy, stream sync |
| **ncu** (Nsight Compute) | 커널 1 회 내부 HW counter | **10-30 × 실시간** (kernel replay) | "이 커널 내부에서 warp 가 매 cycle 뭐 하고 있나" — SOL, 메모리, stall reason, tensor pipe |

이 세션에서 세 번의 다른 용도로 씀:
- **Phase 1**: `nsys` → 시간축에 H2D / kernel / D2H 를 깔고 누가 critical path 인지 본다. **커널끼리 비교가 아니라 커널 + 전송의 균형을 본다.**
- **Phase 2**: `ncu` → 같은 알고리즘 (reduction) 두 구현이 왜 218× 차이가 나는지 **stall reason** 으로 분해.
- **Phase 3**: `ncu` → 우리 커널과 레퍼런스 커널 (SDPA) 을 **같은 shape 에서 동시에 떠서 metric 을 나란히 놓는다**. 20 % gap 이 어디에 있는지 pin-point.

### Phase 0 — 부트스트랩

- 새 VM 생성 (`scripts/gcp_create_l4_spot_vm.sh`, us-west1-b).
- `nsys --version` → 2025.1.3.
- `ncu --version` → 2025.2.
- **GCP Deep Learning 이미지 기본 설정**: `NVreg_RestrictProfilingToAdminUsers=1` — 일반 user 는 `ncu` 만 돌려도 `ERR_NVGPUCTRPERM`. 해결: `sudo -E env PATH=$PATH ncu ...` 로 감싸서 실행 (드라이버 파라미터 flip 하면 재부팅 필요한데 그건 overkill).

### Phase 1 — `nsys` timeline, lesson 04 pageable vs pinned

워크로드: `bin/vector_add --n 16777216 --iterations 5` (64 MB fp32 × 3 array).

[scripts/gcp_run_lesson10_phase1.sh](/Users/xavier/dev/cudatraining/scripts/gcp_run_lesson10_phase1.sh:1): 두 모드 각각 `nsys profile --trace=cuda,nvtx` 뜨고, `nsys stats --report cuda_gpu_mem_time_sum --report cuda_gpu_mem_size_sum` 로 CLI 요약도 추출. `.nsys-rep` 를 로컬로 scp 해서 GUI 에서 열 수 있게 함.

핵심 발견 ([results/lesson10_phase1/summary.md](/Users/xavier/dev/cudatraining/results/lesson10_phase1/summary.md:1)):

| direction | pageable GB/s | pinned GB/s | speedup |
|-----------|---------------|-------------|---------|
| H2D (134 MB)  | 4.77 | 12.35 | 2.59× |
| D2H (67 MB)   | **1.33** | **13.19** | **9.91×** |

**D2H pageable 이 H2D pageable 보다 3.6× 더 느림** (같은 크기 기준). 이유: pageable D2H 는 device → pinned bounce buffer → pageable host 로 2-hop, 마지막 memcpy-to-pageable 가 dominant. H2D 는 reverse 라 이 tail 이 없음. 그래서 pinning 했을 때 D2H 쪽에서 10× 가 한 번에 터지고, 이게 wall-time 에서 가장 눈에 띄게 보인다.

커널 시간은 pageable 0.834 ms / pinned 0.836 ms — **바뀌지 않음**. pinning 은 전송 경로만 건드리지 on-device 계산과 무관하다는 것을 **숫자로** 확인.

### Phase 2 — `ncu` stall diff, lesson 02 reduction v1 vs v4

워크로드: `bin/reduction --n 4194304 --version {1,4}` (4 M elements, 1 launch).

[scripts/gcp_run_lesson10_phase2.sh](/Users/xavier/dev/cudatraining/scripts/gcp_run_lesson10_phase2.sh:1): `ncu --set detailed --launch-skip 20 --launch-count 1 -k "regex:reduce_v{1,4}_" --export reduction_v{1,4}`. 이후 stall metric 을 직접 파싱 (`smsp__pcsamp_warps_issue_stalled_*`).

핵심 발견 ([results/lesson10_phase2/summary.md](/Users/xavier/dev/cudatraining/results/lesson10_phase2/summary.md:1)):

| metric | v1 (atomicAdd per thread) | v4 (warp shuffle + 1 atomic/block) |
|---|---|---|
| Elapsed cycles | **12,085,435** | **55,229** (218× 적음) |
| DRAM throughput | **0.46 %** | **88.21 %** (192× 많음) |
| L2 hit rate | **88.74 %** | 0.95 % |
| Achieved occupancy | 91.17 % | 91.89 % (**같음**) |
| Dominant stall | `lg_throttle` 31.1 % | `long_scoreboard` 84.6 % |

**요지**:
- v1 은 메모리 대역폭이 병목이 아님. DRAM 0.46 % 만 쓰고 있음. 대신 L2 에 hot 하게 박혀 있는 accumulator 라인 (88.7 % hit) 을 전 SM 이 두고 다투는 **atomic serialization** 이 진짜 병목.
- 이게 **`lg_throttle` (31.1 %)** 로 나옴 — load/store unit 의 atomic path 가 back-pressured.
- **v1 과 v4 의 occupancy 가 거의 같음** (91 % vs 92 %). 218× 차이는 occupancy 가 아니라 **"warps 가 매 cycle 뭘 하고 있느냐"** 에서 옴. warp 가 load-modify-store atomic 서로 기다리느라 DRAM 이 놀고 있음.
- v4 는 정상적인 memory-bound 모양: `long_scoreboard` 85 % (DRAM load 기다림), DRAM 88 %, L2 hit 0.95 % (스트리밍 read). `lg_throttle` 0 %.

이 세션 전까지 "atomic 이 느리다" 는 말로만 알고 있었음. stall reason 을 뽑아보니 **atomic 의 "느림" 이 실제로 어떤 HW counter 로 나오는지** 구체적 — `lg_throttle` + L2 hit rate paradox (hit 이 높은데 throughput 이 낮은 기이한 모양).

### Phase 3 — `ncu` diff, ours vs SDPA (lesson 09 의 20 % gap)

워크로드: `B=1 H=32 N=2048 d=128 causal fp16` (LLaMA-7B mid-range shape). 레슨 09 P3 에서 `ours/sdpa = 0.78×` 로 가장 gap 이 큰 shape 이었음.

[triton_kernels/bench/lesson10_phase3_profile.py](/Users/xavier/dev/cudatraining/triton_kernels/bench/lesson10_phase3_profile.py:1): 위 shape 에서 `ours` 또는 `sdpa` 하나만 warmup 20 + iteration 1 로 돌리는 미니 드라이버. `--list-kernels` 모드는 `torch.profiler` 로 실제 dispatch 되는 커널 이름 출력.

[scripts/gcp_run_lesson10_phase3.sh](/Users/xavier/dev/cudatraining/scripts/gcp_run_lesson10_phase3.sh:1): 두 모드 각각 `ncu --launch-skip 20 --launch-count 1 -k <regex>` 로 steady-state 한 번만 캡처. ours 는 `flash_attention_mha`, sdpa 는 `flash|fmha|attention|sdpa|cudnn` regex 로 필터.

**SDPA 가 부른 커널이 cuDNN 이 아니었음** — PyTorch 2.11 의 SDPA 가 L4 + fp16 + causal 조합에서 고른 backend 는 **Tri Dao 의 Flash Attention 2** (`flash_fwd_kernel<Flash_fwd_kernel_traits<128, 64, 64, 4, half_t, ...>>`). 즉 우리는 Triton FA vs **같은 알고리즘의 CUDA/C++ 구현** 을 비교하는 것. 순수 구현 gap.

핵심 발견 ([results/lesson10_phase3/summary.md](/Users/xavier/dev/cudatraining/results/lesson10_phase3/summary.md:1)):

| metric | ours (Triton) | SDPA (FA-2 CUDA) | ratio |
|---|---|---|---|
| Elapsed cycles | 1,565,141 | 827,328 | 1.89× |
| Compute (SM) throughput | 39.3 % | **72.1 %** | 1.84× |
| Tensor pipe utilization | 44.6 % | **78.8 %** | 1.77× |
| Executed IPC | 0.76 | 0.93 | 1.22× |
| DRAM throughput | 10.6 % | 20.3 % | 1.92× |
| **Registers / thread** | **255 (max)** | **184** | 0.72× |
| **Achieved occupancy** | **8.3 %** | **16.2 %** | 1.95× |
| Grid size | 512 | 1024 | 2× |
| Block size | 128 | 128 | = |

Stall 분해:

| stall | ours | SDPA |
|---|---|---|
| total samples | 78,144 | 42,886 |
| `wait` (fixed-latency MMA dep) | **38.6 %** | 19.0 % |
| `selected` (실제 issue) | 21.7 % | 13.6 % |
| `math_pipe_throttle` (tensor saturation) | 19.4 % | **41.5 %** |
| `short_scoreboard` (register dep) | **14.9 %** | 2.2 % |
| `barrier` | 1.1 % | 6.0 % |

**20 % gap 이 있는 곳, 4 가지**:

1. **Register pressure (`255 regs/thread`) → Occupancy 절반**.
   Triton autotune 이 이 shape 에서 `BLOCK_M=128` 을 골랐는데, d=128 head dim 과 겹쳐서 register 가 255 로 spill 직전. SDPA 의 FA-2 는 `BLOCK_M=64, BLOCK_N=64, num_warps=4` 로 184 regs → 2× 더 많은 warp resident.
2. **MMA 파이프라이닝 부족 (`wait` 38.6 %)**. `tl.dot` 직후 output accumulator 를 너무 가까이 consume. `num_stages` 를 3→4 로 올리거나 accumulator 사용 패턴을 재구성하면 줄어들 여지.
3. **Register dependency (`short_scoreboard` 14.9 % — SDPA 는 2.2 %)**. #1 의 연쇄 효과 — register file 이 꽉 차서 producer-consumer 가 자주 같은 물리 register 를 쓰게 됨.
4. **SDPA 는 healthy 한 bottleneck 에 가 있음** — `math_pipe_throttle` 41.5 %, tensor pipe 79 %. "FLOP 을 더 박아야 빨라지는" 구간. 우리는 거기에 도달하지 못함.

실제 follow-up 은 레슨 10 범위 밖이지만 기록:
- autotune 에 `BLOCK_M=64, BLOCK_N=64, num_warps=4` 를 추가해서 SDPA 의 tile 모양을 모방
- `num_stages=4` 시도
- Triton 3.6 의 `tl.dot(acc=)` persistent accumulator 활용

### Phase 4 — 이 문서 + 블로그

각 Phase 별로 `results/lesson10_phase{1,2,3}/summary.md` 에 **숫자 테이블 + interpretation** 을 박아둠. 이 핸드오프는 그 위에 올라가는 서사. 블로그 초안 [docs/blog_draft_lesson_10_profiling.md](/Users/xavier/dev/cudatraining/docs/blog_draft_lesson_10_profiling.md:1) 가 동시 작성.

## 2. 산출물

드라이버:
- [triton_kernels/bench/lesson10_phase3_profile.py](/Users/xavier/dev/cudatraining/triton_kernels/bench/lesson10_phase3_profile.py:1) — Phase 3 ncu 미니 드라이버 (ours / sdpa 단일 mode + kernel 이름 listing)

GCP runner:
- [scripts/gcp_run_lesson10_phase1.sh](/Users/xavier/dev/cudatraining/scripts/gcp_run_lesson10_phase1.sh:1) — nsys timeline (pinned vs pageable)
- [scripts/gcp_run_lesson10_phase2.sh](/Users/xavier/dev/cudatraining/scripts/gcp_run_lesson10_phase2.sh:1) — ncu reduction v1/v4
- [scripts/gcp_run_lesson10_phase3.sh](/Users/xavier/dev/cudatraining/scripts/gcp_run_lesson10_phase3.sh:1) — ncu ours vs SDPA

프로파일링 결과 (scp 로 로컬 pull 됨, GUI 열람 가능):
- [results/lesson10_phase1/vector_add_{pageable,pinned}.nsys-rep](/Users/xavier/dev/cudatraining/results/lesson10_phase1/)
- [results/lesson10_phase2/reduction_{v1,v4}.ncu-rep](/Users/xavier/dev/cudatraining/results/lesson10_phase2/) + `stall_{v1,v4}.json`
- [results/lesson10_phase3/fa_{ours,sdpa}.ncu-rep](/Users/xavier/dev/cudatraining/results/lesson10_phase3/) + `stall_{ours,sdpa}.json`

Phase 별 요약 (표 + 해석):
- [results/lesson10_phase1/summary.md](/Users/xavier/dev/cudatraining/results/lesson10_phase1/summary.md:1)
- [results/lesson10_phase2/summary.md](/Users/xavier/dev/cudatraining/results/lesson10_phase2/summary.md:1)
- [results/lesson10_phase3/summary.md](/Users/xavier/dev/cudatraining/results/lesson10_phase3/summary.md:1)

로그:
- `results/lesson10-phase{1,2,3}-*.log`

문서:
- 이 파일 — 핸드오프
- [docs/blog_draft_lesson_10_profiling.md](/Users/xavier/dev/cudatraining/docs/blog_draft_lesson_10_profiling.md:1) — 블로그 초안

메타:
- [README.md](/Users/xavier/dev/cudatraining/README.md:1) 에 레슨 10 섹션 추가 예정

## 3. 핵심 숫자 (L4 sm_89, CUDA 12.9, nsys 2025.1.3, ncu 2025.2)

### 3.1 Phase 1 — pinned vs pageable (lesson 04)

| direction | pageable GB/s | pinned GB/s | 커널 영향 |
|-----------|---------------|-------------|-----------|
| H2D       | 4.77          | 12.35       | 없음 (0.834 ms 고정) |
| D2H       | **1.33**      | **13.19**   | 없음 |

Wall-time D2H: pageable 51.9 ms → pinned 5.1 ms (**10 × 개선**).

### 3.2 Phase 2 — reduction v1 vs v4 (lesson 02)

|                  | v1    | v4    |
|------------------|-------|-------|
| Elapsed cycles   | 12.1 M | 55 K (218× 적음) |
| DRAM throughput  | 0.46 % | 88.2 % |
| L2 hit           | 88.7 % (hot line!) | 0.95 % |
| Occupancy        | 91 %  | 92 %  |
| `lg_throttle`    | 31 %  | 0 %   |

### 3.3 Phase 3 — ours vs SDPA (lesson 09, `B=1 H=32 N=2048 d=128 causal fp16`)

|                   | ours   | SDPA  |
|-------------------|--------|-------|
| Elapsed cycles    | 1.57 M | 0.83 M (1.89× 적음) |
| Tensor pipe util  | 44.6 % | 78.8 % |
| Compute SM %      | 39.3 % | 72.1 % |
| DRAM %            | 10.6 % | 20.3 % |
| Regs / thread     | 255    | 184 |
| Occupancy         | 8.3 %  | 16.2 % |
| `wait` stall      | 38.6 % | 19.0 % |
| `math_pipe_throttle` | 19.4 % | **41.5 %** |

## 4. 세 가지 교훈

### (a) **Occupancy 는 throughput 이 아니다**

Phase 2 의 가장 비직관적인 모양: v1 과 v4 둘 다 occupancy 91-92 % 로 같은데 wall cycle 은 218× 차이. occupancy 는 "몇 warp 가 동시에 살아있을 수 있는가" 이지 "그 warp 들이 cycle 당 뭘 하고 있는가" 가 아니다.

v1 의 90 % occupancy 는 **90 % 의 warp 가 atomic 을 기다리고 있음** 을 뜻한다. 쓰임새 없이 앉아 있는 warp 가 많아도 occupancy 지표는 올라간다.

맞는 지표 체크 순서:
1. **DRAM throughput %** — memory-bound 인가?
2. **Compute (SM) throughput %** — compute-bound 인가?
3. 둘 다 낮으면 → **stall reason** 을 본다. 뭔가가 warp 를 막고 있다.

### (b) **`ncu` stall metric 은 warp 에게 "cycle 마다 뭐 했어?" 를 물어본다**

각 cycle 의 각 warp 는 여덟 중 하나의 상태:
- `selected` — 이 cycle 에 명령어 issue 한 warp (productive)
- `not_selected` — 발사 준비는 됐는데 스케줄러가 다른 warp 를 골랐음
- `stalled_<reason>` — 다음 중 하나 때문에 못 뜸:
  - `wait` — fixed-latency pipeline (MMA, SFU) 결과 대기
  - `long_scoreboard` — global/local memory load/store 대기 (L2/DRAM)
  - `short_scoreboard` — shared memory / register dep
  - `lg_throttle` — LSU throttle (atomics, misaligned access 등)
  - `math_pipe_throttle` — tensor/FP pipe 가 꽉 차서 내가 못 들어감 (*좋은 stall*)
  - `barrier` — `__syncthreads` / `bar.sync`
  - `mio_throttle` — memory IO unit
  - `drain` — warp 끝날 때 in-flight op 대기

이 분포가 바뀌면 kernel 의 성격이 바뀐 것. 같은 알고리즘의 다른 구현 (Phase 2: v1 vs v4; Phase 3: ours vs SDPA) 을 비교할 때 **stall 분포의 shape 를 보면 두 구현이 뭘 다르게 하고 있는지** 보인다.

`math_pipe_throttle` dominant = **healthy** — 산수를 더 박아야 빨라짐.
`wait` dominant = MMA output dependency chain — 소프트웨어 파이프라이닝 더 필요.
`lg_throttle` dominant = atomics / strided access — 알고리즘 재설계 신호.

### (c) **작은 tile 이 빠를 수 있다 — occupancy 때문**

Phase 3 의 표면적 역설: SDPA 는 `BLOCK_M=64 BLOCK_N=64` (작은 tile), 우리는 autotune 이 `BLOCK_M=128` (큰 tile) 을 골랐는데 SDPA 가 더 빠름.

직관: 큰 tile → 각 block 이 하는 일이 많음 → "efficient" 할 것 같음.
현실: 큰 tile → register 많이 씀 → **resident warp 수가 줄어듦** → SM 이 warp 돌림판 돌리다가 비어버림.

**255 regs/thread** 는 spill 직전 임계. Triton 이 이 config 를 골라준 건 한 warp 의 계산 밀도가 높아서지만, SM 전체 관점에서는 **warp pool** 이 고갈됨. SDPA 의 184 regs/thread 는 2 배의 warp 를 동시에 살려 둘 수 있음.

Autotune 이 wall time 기반으로 best config 를 골라도, **해당 shape 에서 "좋은 bottleneck" 에 도달하지 못했을 수 있다**. Phase 3 에서는 ours 가 `wait` (MMA 의존성) 로 막혀 있어서 tensor core 에 꽂혀서 throttle 되는 지점까지 못 감.

actionable: 레슨 09 autotune config pool 에 "작은 tile + 많은 stage" 조합 (`BLOCK_M=64, BLOCK_N=64, num_stages=4, num_warps=4`) 을 추가해보는 것. 레슨 10 범위 밖이지만 바로 다음 iteration 후보.

## 5. 함정 기록

### 함정 1: L4 stockout 이 이제 us-east4-c 에서도 안 먹힘

레슨 09 핸드오프에서 "us-east4-c 가 L4 재고가 잘 도는 편" 이라고 적어뒀는데, 이번 세션에서는 us-east4-c 도 안 됨. zone rotation 순서:

```
us-east4-c (✗) → us-central1-a (✗) → us-central1-b (✗) → us-central1-c (✗)
→ us-west1-a (✗) → us-west1-b (✓)
```

**다음 세션 시작할 때 zone rotation 은 당연한 절차로 잡자.**

### 함정 2: GCP DL 이미지 + ncu → `ERR_NVGPUCTRPERM`

증상:
```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to profile on the target device.
```

원인: `cat /proc/driver/nvidia/params | grep Restrict` → `NVreg_RestrictProfilingToAdminUsers: 1`. 일반 user 는 perf counter 에 접근 불가. GCP DL 이미지 기본 설정.

해결:
```bash
NCU="sudo -E env PATH=$PATH ncu"
$NCU --set detailed --launch-count 1 -k "regex:..." --export out ./bin
```

드라이버 파라미터를 바꾸려면 `/etc/modprobe.d/*` 수정 + 재부팅 + 드라이버 재로드 — overkill. `sudo` wrapper 가 충분.

`sudo -E env PATH=$PATH` 가 중요 — `sudo ncu` 만 하면 PATH 에 `/usr/local/cuda/bin` 안 들어가서 `ncu` 찾지 못하는 경우 있음.

### 함정 3: ncu CSV 파싱 - row 2 는 values 아님, **units**

`ncu --import x.ncu-rep --csv --page raw` 의 row 구조:
- row 1: metric 이름 헤더
- row 2: **단위 (`"warp"`, `"inst"`, `"%"`, `""`)**
- row 3+: 실제 값 (kernel launch 당 한 row)

처음에 awk 로 row 2 를 읽어서 값이 다 `"warp"` 로 나와서 한참 헤맴. Python `csv.reader` + `rows[2]` 가 정답.

csv 필드가 따옴표 처리 / 공백 등 pathological 하므로 awk/cut 보다는 Python csv module 이 가장 안전.

### 함정 4: `torch.profiler` API 가 버전마다 다름 (torch 2.11)

레슨 09 때는 `cuda_time_total` 로 됐는데 torch 2.11 에선 `FunctionEventAvg` 에 이 attribute 가 없음 → `device_time_total` 로 바뀜 (MPS / ROCm 지원을 위한 generic 이름).

방어적으로:
```python
def dev_time(e):
    for attr in ("device_time_total", "cuda_time_total", "self_device_time_total"):
        if hasattr(e, attr):
            return getattr(e, attr)
    return 0.0
```

### 함정 5: `ncu` replay 는 10-30 × 실시간

`--set detailed` 는 17 passes (Phase 3 기준). 커널 하나가 1 ms 라도 **ncu 에서 한 번 profile 에 수 초 ~ 수십 초**. 이유: 각 pass 에서 서로 다른 HW counter 집합을 수집해야 해서 커널을 **여러 번 replay**.

대책:
- `--launch-count 1` — 한 번의 launch 만 뜨기
- `--set base` (가벼운 metric set) — pass 수 감소. 단 이 세션에서 본 stall / tensor pipe 는 안 나올 수 있음. `detailed` 가 현실적 최소.
- 큰 모델 / 긴 seq 에서 느릴 때는 **shape 을 줄이고** 프로파일 → 패턴은 유지됨.

### 함정 6: SDPA 의 backend 를 모른 채로 비교하는 함정

처음엔 "SDPA = cuDNN flash attention" 이라고 생각하고 Phase 3 를 시작했는데, kernel 이름을 뽑아보니 `flash_fwd_kernel<Flash_fwd_kernel_traits<...>>` — **Tri Dao 의 FA-2 CUDA 구현이 torch 에 번들** 된 것. cuDNN 이 아님.

backend 힌트:
```python
# 특정 backend 만 활성화하고 싶으면:
with torch.nn.attention.sdpa_kernel(
    [torch.nn.attention.SDPBackend.FLASH_ATTENTION,
     torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
     torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]
):
    out = F.scaled_dot_product_attention(...)
```

레슨 10 에서 별도로 cuDNN vs FA-2 비교는 안 했지만, **"SDPA 를 이겼다 / 졌다" 라고 말할 때는 어떤 backend 와 비교하는 건지 기록** 필요.

## 6. 다음 세션에 남기는 것

- Phase 3 의 actionable 4 개 (register pressure 완화, `num_stages=4`, 작은 tile autotune 추가, `tl.dot(acc=)`) 를 **레슨 11 (또는 레슨 09 patch)** 로 실험해볼 후보.
- `nsys` / `ncu` 커맨드 + 스크립트 템플릿이 잡혔으니 다음 신규 커널 (matmul 관련, attention 변형 등) 은 처음부터 profile 붙여서 설계 가능.
- GPU 를 바꿔서 (예: T4, A10) 같은 ncu 를 돌리면 **GPU 간 bottleneck 이 어떻게 이동하는지** 볼 수 있음 — L4 에서 tensor-bound 인 shape 이 T4 에선 memory-bound 가 될 것.

## 7. 요약 한 문단

> 레슨 1-9 에서 "빠르다/느리다" 로 끝났던 세 개의 주장을 nsys / ncu 로 숫자화했다.
> (1) 레슨 04 의 pinned 효과는 D2H 에서 **10×**, H2D 에서 **2.6×** — pageable D2H 가 왜 유독
> 느린지가 `nsys` 의 timeline 에서 2-hop memcpy 로 보인다.
> (2) 레슨 02 의 reduction v1→v4 의 218× 차이는 **occupancy 가 아니라** atomic serialization
> (`lg_throttle` 31 %) 때문 — 두 구현의 occupancy 는 같다.
> (3) 레슨 09 의 ours = SDPA 의 78-90 % 에서 잃는 **20 %** 는 register pressure (255 regs, occupancy
> 절반) + MMA wait (`wait` 39 %) 에 있음. SDPA 는 이미 tensor pipe throttle (healthy bottleneck)
> 까지 가 있음.
> 이 세 인사이트는 **프로파일링 툴 없이는 말이 되지 않는 해석** 이고, 이것이 lesson 10 가
> 정적 벤치와 구분되는 지점.
