# Lesson 11 · Phase 3 — Paged Attention Speed + Profile

**Date**: 2026-04-22
**GPU**: NVIDIA L4 (sm_89, 300 GB/s DRAM, 24 GB)
**Stack**: torch 2.11.0+cu128, Triton 3.6.0
**dtype**: fp16, warmup=50, iters=200

Baseline = `torch.nn.functional.scaled_dot_product_attention(..., enable_gqa=True)`
— the dispatcher picks cuDNN / FA-2 / aten.

---

## Headline

**MHA: parity with SDPA.** Our paged kernel matches or beats SDPA at
realistic batch sizes, peaking at **253 GB/s** (≈ 85 % of L4 DRAM peak).

**GQA: structurally 2×–13× slower than SDPA.** Root cause is the
`(B, H_q)` grid launching one program per query head, so the
`GQA_GROUP_SIZE` query heads in a KV group each independently reload the
**same** KV blocks. vLLM's actual kernel launches `(B, H_kv)` and
broadcasts Q inside the kernel — that's the fix for Phase 4.

---

## Full table

```
| shape                  | B  | H  | H_kv | ctx  | SDPA ms | SDPA GB/s | bs=8 x | bs=16 x | bs=32 x | bs=64 x | bs=128 x |
|------------------------|----|----|------|------|---------|-----------|--------|---------|---------|---------|----------|
| llama7b-B1-ctx1k       | 1  | 32 | 32   | 1024 | 0.035   | 485.4     | 0.23x  | 0.42x   | 0.71x   | 0.71x   | 0.73x    |
| llama7b-B1-ctx4k       | 1  | 32 | 32   | 4096 | 0.263   | 254.8     | 0.32x  | 0.60x   | 0.93x   | 0.93x   | 0.96x    |
| llama7b-B8-ctx2k       | 8  | 32 | 32   | 2048 | 1.143   | 235.0     | 0.98x  | 1.08x   | 1.08x   | 1.02x   | 1.08x    |
| llama7b-B32-ctx2k      | 32 | 32 | 32   | 2048 | 4.254   | 252.5     | 0.94x  | 0.92x   | 1.00x   | 1.00x   | 1.00x    |
| llama7b-B8-ctx8k       | 8  | 32 | 32   | 8192 | 4.683   | 229.3     | 0.97x  | 1.10x   | 1.10x   | 1.04x   | 1.10x    |
| llama38b-B8-ctx2k      | 8  | 32 | 8    | 2048 | 0.271   | 248.3     | 0.29x  | 0.46x   | 0.73x   | 0.80x   | 0.86x    |
| llama38b-B32-ctx2k     | 32 | 32 | 8    | 2048 | 1.147   | 234.5     | 0.31x  | 0.50x   | 0.71x   | 0.70x   | 0.73x    |
| llama70b-B4-ctx2k      | 4  | 64 | 8    | 2048 | 0.069   | 488.3     | 0.08x  | 0.14x   | 0.24x   | 0.30x   | 0.31x    |
| llama70b-B8-ctx4k      | 8  | 64 | 8    | 4096 | 0.533   | 252.2     | 0.16x  | 0.25x   | 0.38x   | 0.46x   | 0.46x    |
| mqa-B16-ctx4k          | 16 | 32 | 1    | 4096 | 0.062   | 542.1     | 0.02x  | 0.03x   | 0.05x   | 0.07x   | 0.07x    |
```

Speedup = SDPA_ms / paged_ms. `> 1.0x` means paged is faster.

---

## Signatures

### Signature 1 — the gap tracks GQA_GROUP_SIZE

```
  group=1 (MHA):       gap ~ -10% to +0%  (parity)
  group=4 (LLaMA-3-8B):  gap ~ +16%
  group=8 (LLaMA-70B):   gap ~ +117%  to +227%
  group=32 (MQA):        gap ~ +1316%
```

The gap scales roughly linearly with GQA_GROUP_SIZE because each query
head in the group performs a separate full scan of the KV cache — `N_q`
redundant copies of the same KV DRAM read.

### Signature 2 — SDPA hits cached L2, ours hits DRAM

- SDPA on MQA B=16 ctx=4k: **542 GB/s** (above L4 DRAM peak of 300 GB/s).
  Impossible from DRAM alone → L2 is absorbing the repeated K/V reads.
  The one KV head's data (16 MB) fits in L2 (48 MB) and the 32 query
  heads reuse it.
- Our paged on the same shape: **38 GB/s**. 14× below SDPA. We
  re-fetch from DRAM per query head.

### Signature 3 — block_size curve

- `bs=8`: always worst (block_table load overhead dominates).
- `bs=16`: good on MHA, weak on GQA.
- `bs=128`: safest default — less block_table traffic per token.

Block size gains flatten above 32 for MHA and above 64 for GQA.

---

## Fix sketch (for Phase 4)

Current:
```python
grid = (B, H_q)                      # one program per (batch, query_head)
kv_head = pid_h // GQA_GROUP_SIZE    # each group redundantly loads KV
```

vLLM-style:
```python
grid = (B, H_kv)                      # one program per (batch, kv_head)
# Inside the program:
#   - Load K/V blocks for this (batch, kv_head) once
#   - Load GQA_GROUP_SIZE Q rows
#   - For each KV block, compute GROUP_SIZE dot products,
#     maintain GROUP_SIZE online-softmax states
#   - Store GROUP_SIZE output rows
```

Memory traffic drops by `GQA_GROUP_SIZE` factor. Register pressure goes
up (group × state), so tl.dot may need `BLOCK_M = GROUP_SIZE` with padding.

---

## Conclusion (three lines)

1. **MHA is shipped.** Our kernel ≈ SDPA at realistic batch, 253 GB/s peak.
2. **GQA needs a grid restructure.** Launch per `(B, H_kv)`, broadcast Q
   across the group inside the kernel. Same fix vLLM already did.
3. **Default block_size = 128 on L4** unless the shape is short-ctx B=1
   (then launch overhead is the bottleneck; block_size is irrelevant).
