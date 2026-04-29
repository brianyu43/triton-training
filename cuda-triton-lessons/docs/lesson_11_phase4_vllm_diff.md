# Lesson 11 · Phase 4 — vLLM source reading + diff notes

**Goal**: Compare my Phase 3.5 Triton paged-attention kernel against
the three canonical paged-attention kernels shipped in vLLM. Validate
where I converged on the right design, catch what I missed, and sketch
the split-k that would close the residual MQA gap.

**Source**: `github.com/vllm-project/vllm` @ main (shallow clone in
`/tmp/vllm` during this session).

Kernels compared:

| # | Kernel | File | Role |
|---|---|---|---|
| v1 | `paged_attention_v1` | `csrc/attention/paged_attention_v1.cu` + `attention_kernels.cuh` | The OG CUDA kernel (2023). Per-query-head grid. |
| v2 | `paged_attention_v2` + `_reduce` | `csrc/attention/paged_attention_v2.cu` | Split-k along ctx, 2-stage launch. |
| tri | `kernel_unified_attention_{2d,3d}` | `vllm/v1/attention/ops/triton_unified_attention.py` | Modern Triton kernel. Per-KV-head grid. Replaces v1 on newer stacks. |

Mine: `triton_kernels/paged_attention.py` (Phase 3.5).

---

## TL;DR

1. **Phase 3.5 convergently reinvented the modern vLLM design.** The
   CUDA v1 kernel launches per-query-head and reuses KV through L2
   alone; the Triton unified kernel launches per-KV-head with BLOCK_M
   query rows per program and uses `tl.dot`. My Phase 3.5 matches
   axis-for-axis the Triton unified design. vLLM itself went through
   the same refactor I did — per-query-head → per-KV-head — when they
   moved from CUDA to Triton.

2. **My IEEE fix is a legitimate-but-unique call.** vLLM never sets
   `input_precision="ieee"` on their `tl.dot`. Production vLLM runs
   fp16/bf16 only, so they don't see the TF32-on-fp32 trap I hit. If
   someone tried fp32 through their Triton path on sm_80+ they'd get
   the same 4e-4 bleed I caught in correctness.

3. **The residual MQA gap is a v2-shaped hole.** vLLM closes it with a
   ctx-axis split-k (`kernel_unified_attention_3d` / `paged_attention_v2`)
   + a separate reduce kernel that recombines per-segment
   max-logits + exp-sums. Phase 4.5 sketch below.

---

## Design-axis diff table

| axis | vLLM CUDA v1 | vLLM CUDA v2 | vLLM Triton unified | **Mine (Phase 3.5)** |
|---|---|---|---|---|
| **grid** | `(H_q, B)` | `(H_q, B, max_partitions)` | `(Σ q_blocks, H_kv)` or `(..., H_kv, segments)` | `(B, H_kv)` |
| **programs per `(B, H_kv)`** | `GROUP_SIZE` (redundant) | `GROUP_SIZE × segments` | `q_len / BLOCK_Q` | **1** |
| **Q tile per program** | 1 query head, `HEAD` elts | 1 query head, `HEAD` elts | `(BLOCK_M, HEAD)` — packs rows | `(GROUP, HEAD)` |
| **K load dedup across group** | relies on L2 | relies on L2 | yes, by construction | yes, by construction |
| **matmul primitive** | warp-level `Qk_dot` w/ shfl | warp-level `Qk_dot` w/ shfl | `tl.dot(Q, K)` | `tl.dot` (GROUP≥4) or manual broadcast |
| **fp32 correctness** | N/A (fp16/bf16 only) | N/A | default TF32 (would bleed) | **IEEE explicit on fp32 path** |
| **softmax state** | block-wide w/ shared mem | block-wide per-segment | per-BLOCK_M row, Triton-level | per-GROUP row, Triton-level |
| **split-k on ctx** | no | yes (`PARTITION_SIZE = 512`) | yes (`segm_idx` axis) | **no** ← MQA residual |
| **K cache layout** | `(NB, H_kv, HEAD/x, BLK, x)` | same | `(NB, BLK, H_kv, HEAD)` | `(NB, BLK, H_kv, HEAD)` ✅ same |
| **V cache layout** | `(NB, H_kv, HEAD, BLK)` | same | `(NB, BLK, H_kv, HEAD)` | `(NB, BLK, H_kv, HEAD)` ✅ same |
| **BLOCK sizes shipped** | 8, 16, 32 | 8, 16, 32 | variable (`block_size` = `v.shape[1]`) | 8, 16, 32, 64, 128 |
| **HEAD sizes shipped** | 32–256 (9 specials) | same | variable, padded to pow2 | 32, 64, 128 |
| **prefill path** | no (decode only) | no (decode only) | **yes** — packs rows via BLOCK_Q | no (decode only, like lesson scope) |
| **TILE_SIZE vs BLOCK_SIZE** | same (= BLOCK) | same | decoupled | same (= BLOCK) |
| **extras** | ALiBi, block-sparse | +partition logic | +sliding window, softcap, sinks, qq-bias, fp8, MM prefix | none (lesson scope) |

Legend: ✅ = same as mine; the table is mostly for reading row-by-row.

---

## Where Phase 3.5 matches vLLM Triton unified (converged)

**A. Grid is (… , H_kv).** See `triton_unified_attention.py:181–182`:

```python
q_block_global_idx = tl.program_id(0)
kv_head_idx = tl.program_id(1)
```

Mine (`paged_attention.py:84–85`):

```python
pid_b = tl.program_id(axis=0)     # batch
pid_kv = tl.program_id(axis=1)    # kv_head
```

vLLM's axis-0 is a merged `(batch × query-block)` index because they
handle variable query lengths (prefill); mine is pure batch because
decode has one query token per sequence. Both axes-1 are `kv_head_idx`.
Same fundamental structure.

**B. Q packed into a 2D tile covering the GQA group.** vLLM
(`triton_unified_attention.py:218–222`):

```python
Q = tl.load(query_ptr + query_offset, mask=..., other=0.0)
# Q : (BLOCK_M, HEAD_SIZE_PADDED)
```

Mine (`paged_attention.py:99`):

```python
q = tl.load(q_ptrs)  # (GROUP, HEAD_DIM)
```

vLLM's `BLOCK_M = 16 if num_queries_per_kv <= 16 else next_pow2(...)`
(line 1078). For GROUP ≥ 16 this matches my `(GROUP, HEAD)`. For GROUP
< 16 vLLM pads by packing more query rows (`BLOCK_Q > 1`) to keep
BLOCK_M ≥ 16 for MMA. My decode-only kernel can't do that (one query
token per sequence), so I fall back to manual broadcast at GROUP < 4.

**C. `tl.dot(Q, K)` for scores, `tl.dot(P, V)` for accumulation.** vLLM
(lines 410, 479):

```python
S += scale * tl.dot(Q, K)
...
acc += tl.dot(P.to(V.dtype), V)
```

Mine (`paged_attention.py:157, 182`):

```python
scores = tl.dot(q_scaled, tl.trans(k)).to(tl.float32)
...
acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
```

Identical shape and fp32-accumulator pattern.

**D. Online softmax running state.** vLLM (lines 227–236, 443–465) and
mine (`paged_attention.py:107–109, 167–172`) both carry `(M, L, acc)`
per-row and apply the standard `alpha = exp(m_old - m_new)` rescale.
Same math.

**E. KV cache layout.** vLLM Triton (lines 120–121):

```python
key_cache_ptr,    # [num_blks, blk_size, num_kv_heads, head_size]
value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
```

Mine (`paged_attention.py:17–18`):

```
K_cache : (num_blocks, block_size, H_kv, d)
V_cache : (num_blocks, block_size, H_kv, d)
```

Same layout, identical dim order. I converged on this independently by
reading the vLLM *paper*, not the modern Triton code.

---

## Where vLLM goes further than I did (production machinery)

1. **Prefill + decode unified kernel.** vLLM launches one kernel for
   both. `BLOCK_Q = BLOCK_M // num_queries_per_kv` lets them pack either
   query rows (prefill, long `q_len`) or batch rows (decode, `q_len=1`)
   into one program. My kernel is decode-only — intentional for the
   lesson, but worth noting.

2. **Split-k via 3rd grid axis.** `kernel_unified_attention_3d` adds a
   `segm_idx = program_id(2)` axis that partitions the ctx dimension.
   Each `(q_block, kv_head, segment)` program writes partial
   max-logits + exp-sums + partial out, and a second reduce kernel
   (`reduce_segments`) folds them. This is what I need to close MQA.
   Blueprint in §"Phase 4.5" below.

3. **Decoupled `BLOCK_SIZE` (paging) vs `TILE_SIZE` (compute).** vLLM
   can load an 8-token paged block but compute on 64-token tiles,
   amortizing the block_table lookup. My `BLOCK_SIZE` is both — which
   is why tiny paging blocks (`bs=8`) show launch overhead in my Phase 3
   numbers.

4. **Production features I don't have**: ALiBi slopes, sliding window,
   softcap (Gemma-2), attention sinks, qq-bias (multimodal), fp8 KV
   cache with per-head scales, MM prefix ranges (vision), block-sparse
   attention. Each is a small cost in the kernel once; all orthogonal
   to the grid design, so Phase 3.5's design accommodates any of them.

5. **CUDA v1 uses manual 16B vectorization** (`VEC_SIZE = 16 /
   (THREAD_GROUP_SIZE * sizeof(scalar_t))`) and explicit warp
   reductions (`VLLM_SHFL_XOR_SYNC`). Triton hides both — `tl.dot` and
   `tl.max` / `tl.sum` get the equivalent ld.global.v4 + warp-shuffle
   at compile time. Trade: less code (mine is 275 lines, their CUDA is
   670), less fine-grained control (mine can't hand-unroll the
   thread_group load).

---

## Where I went further than vLLM

1. **Explicit fp32 IEEE precision.** vLLM never sets
   `input_precision="ieee"` — their `tl.dot` defaults to TF32 on
   Ampere+. For their production dtype (fp16/bf16) this doesn't
   matter. For fp32 it would bleed ~4e-4, which the Phase 1+2 bench
   caught on my MQA case. My kernel branches on `IS_FP32` and flips
   IEEE only where needed.

2. **Phase 3 speed bench + block-size sweep with explicit gap
   reporting.** vLLM has kernel-level benchmarks in the repo but not
   a readable "SDPA vs ours at these 10 shapes × 5 block sizes" table
   wired as a smoke test. The speed bench (`bench_paged_attention_speed.py`)
   is mine.

3. **Decode-focused clarity.** The lesson kernel is ~275 lines; the
   CUDA v1 is 670 in the kernel body + 186 in the launcher + 57 in
   utils. Easier to reason about correctness for learning purposes. Not
   a production advantage, just a teaching one.

---

## The CUDA v1 vs Triton unified "refactor"

This is the most interesting historical pattern. vLLM shipped
`paged_attention_v1.cu` in 2023 with `dim3 grid(num_heads, num_seqs)`
— i.e., one program per query head, which is exactly my Phase 3
design (the slow one). Every query head in a GQA group reloaded the
same KV blocks, and they relied entirely on L2 to hide the redundant
DRAM traffic. This worked because:

- At the time, most models were MHA (H_kv == H_q), so the "GQA group
  redundancy" problem literally didn't exist — each query head had its
  own KV anyway.
- For the few GQA models (MQA in PaLM), the KV footprint was small
  enough to fit in L2 comfortably.
- Triton 1.x/2.0 in 2023 couldn't reliably generate competitive MMA
  for variable tile sizes; CUDA with manual vector loads and warp
  shuffles was the path to performance.

When Triton matured (3.x, MMA + shared-mem management improved) and
LLaMA-2-7B-chat, LLaMA-3, Mistral all shipped with GQA, the
per-query-head grid became the bottleneck. vLLM moved the kernel to
Triton and restructured the grid to `(q_block, H_kv)` — the **same
refactor I did in Phase 3.5**, albeit driven by the same forcing
function (GQA shapes where group size ≥ 4).

The PR that introduced `triton_unified_attention.py` is in the vLLM
history around the v1 → v1-stable transition. Not worth chasing the
exact PR; the design direction is unambiguous from the source as it
stands today.

**Takeaway for the story**: this isn't me doing a clever thing vLLM
never thought of. This is me independently arriving at the same
design vLLM already concluded was correct. Convergence is the
signal. "The ecosystem already did this refactor, here's the
same bug and the same fix reproduced in miniature" is a credible
kernel-engineering story.

---

## Phase 4.5 blueprint — closing the MQA gap with split-k

### Problem recap

MQA on L4: SDPA 698 GB/s (2.3× DRAM peak) vs our paged 89 GB/s at
`bs=128`. The grid fix didn't close this because there's only 1 KV
head and 1 batch → we launch **16 programs total** (one per `(b, h_kv)`
at B=16, H_kv=1), which leaves 42 of L4's 58 SMs idle, and each
program must walk 4096 / 128 = 32 blocks serially.

SDPA wins because it launches 16×32 = 512 programs (per query head)
and uses L2 to absorb the tiny 1 MB KV footprint reuse. Our paged
layout breaks the L2 prefetch pattern for the block_table indirection.

### Fix: split the ctx axis

```
grid = (B, H_kv, SEGMENTS)           # axis 2 is new
SEGMENTS = ceil(ctx / PARTITION_SIZE)  # e.g. 512-token segments
```

Each program handles `PARTITION_SIZE` tokens (= `PARTITION_SIZE / BLOCK_SIZE`
paged blocks) and writes:

- `segm_max[b, h_q, s]`  : max logit in this segment
- `segm_expsum[b, h_q, s]` : sum_j exp(S_j - segm_max) in this segment
- `segm_out[b, h_q, s, d]` : ∑_j exp(S_j - segm_max) * V_j (unnormalized)

A second reduce kernel over `(B, H_kv)` then:

```python
M_global = max over s of segm_max[b, h_q, s]
for s in segments:
    alpha_s = exp(segm_max[b, h_q, s] - M_global)
    L_global += alpha_s * segm_expsum[b, h_q, s]
    out_global += alpha_s * segm_out[b, h_q, s, :]
out[b, h_q] = out_global / L_global
```

This is exactly vLLM's v2. The launcher picks `PARTITION_SIZE = 512`
(v2) — enough to keep each program's work non-trivial, small enough
that MQA/B=16 gets `1 × 1 × 8 = 8` programs on the forward pass and
`16 × 1 = 16` on reduce, still more SMs than before but not linearly
proportional to ctx_len.

### Expected L4 impact

- MQA B=16 ctx=4k: 16 programs → ~128 programs (partition=512,
  segments=8). Now 128 programs on 58 SMs = 2.2 waves, competitive
  with SDPA.
- The L2 reuse story is roughly preserved because each segment of the
  single KV head (128 KB segment) fits in L2 and is reused by the 32
  query heads through the GQA group.

### Cost

- Two kernel launches instead of one (+~15 µs on L4 for the launch).
- Scratch memory: `B * H_q * SEGMENTS * (1 float + 1 float + HEAD_DIM * element_size)`
  = 16 × 32 × 8 × (8 + 128 × 2) = ~1.1 MB. Fine.
- Reduce kernel is arithmetically trivial; bandwidth-bound on the
  `segm_out` read.

### When NOT to split

The reduce kernel has launch overhead (~15 µs on L4) that only pays
back when the forward pass is ≥ ~30 µs. For short ctx or already-
saturated grids (MHA at large B), the split is a regression. vLLM's
launcher picks split-k conditionally on `num_par_softmax_segments`
and `seq_threshold_3D`; mine would do the same.

### Implementation order (Phase 4.5, if I do it)

1. Add `SEGMENTS: tl.constexpr` + `segm_idx = program_id(axis=2)` to
   the decode kernel, partition the block loop.
2. Write the reduce kernel (~50 lines Triton, no tl.dot).
3. Wrap: if `SEGMENTS == 1` just call forward; else forward +
   reduce.
4. Correctness bench re-run (32/32 should pass).
5. Speed bench re-run — target MQA gap < +20% (reasonable; the L2
   reuse ceiling is structural).

Estimate: **half a day**. Not on the critical path to shipping
Lesson 11 — the GQA story is already complete — but a clean lead-in
to Lesson 12.

---

## Final ledger

| question | answer |
|---|---|
| Did I reinvent the right design? | **Yes.** My Phase 3.5 grid matches vLLM Triton unified axis-for-axis. |
| Did I miss something vLLM knew? | **One thing.** Split-k over ctx (v2) closes the MQA gap; I don't have it. |
| Did I add something vLLM doesn't have? | **One thing, narrowly.** Explicit `input_precision="ieee"` on fp32 tl.dot. Irrelevant to production (fp16/bf16), relevant to a lesson that tests fp32. |
| Should I do Phase 4.5 before Lesson 11 ships? | **Optional.** The story is complete without it. Nice-to-have. |

Source files I read (all `/tmp/vllm/...`):
- `csrc/attention/paged_attention_v1.cu` (186 lines)
- `csrc/attention/paged_attention_v2.cu` (196 lines)
- `csrc/attention/attention_kernels.cuh` (670 lines — the real kernel)
- `csrc/attention/attention_utils.cuh` (57 lines — Qk_dot helper)
- `vllm/v1/attention/ops/triton_unified_attention.py` (1268 lines — modern Triton replacement)
