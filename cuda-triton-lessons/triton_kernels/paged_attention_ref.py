"""
Lesson 11 · Phase 0 — Paged Attention PyTorch reference.

Correctness oracle for the Triton paged attention kernel. Implements the
vLLM-style paged KV cache layout and runs standard attention through an
explicit gather step (block_table → physical blocks → contiguous K/V).

Not fast. Not intended to be. The point is to have something we can
`torch.allclose` against when the Triton kernel is written.

Data layout (vLLM-ish, simplified — no vectorization reshape yet):

    K_cache: (num_blocks_total, block_size, H_kv, d)
    V_cache: (num_blocks_total, block_size, H_kv, d)
    block_table: (B, max_blocks_per_seq) int32
        block_table[b, i] = physical block id for the i-th logical block of
        sequence b. Blocks past context_lens[b] are unused (padding).
    context_lens: (B,) int32
        Number of valid tokens in each sequence's KV history.

Decode convention:
    Q: (B, H, d) — one query token per sequence (the new token being
    generated). No separate N dimension. This mirrors how vLLM calls
    the kernel during autoregressive decoding.
"""

from __future__ import annotations

import torch


def pack_kv_paged(
    K: torch.Tensor,
    V: torch.Tensor,
    block_size: int,
    context_lens: torch.Tensor | None = None,
):
    """Convert contiguous (B, H, N, d) KV tensors to paged format.

    Allocates one physical block per logical block, sequentially. No block
    sharing between sequences. Good enough as a correctness oracle.

    Args:
        K, V: (B, H, N, d), same dtype / device.
        block_size: tokens per block.
        context_lens: (B,) int. If None, assumes all sequences have length N.

    Returns:
        K_cache, V_cache: (num_blocks_total, block_size, H, d).
            Unused slots at the end of partial last blocks are zeroed.
        block_table: (B, max_blocks_per_seq) int32.
        context_lens: (B,) int32.
    """
    assert K.shape == V.shape, f"K {K.shape} != V {V.shape}"
    B, H, N, d = K.shape
    device = K.device

    if context_lens is None:
        context_lens = torch.full((B,), N, dtype=torch.int32, device=device)
    else:
        context_lens = context_lens.to(dtype=torch.int32, device=device)
        assert context_lens.shape == (B,)
        assert int(context_lens.max().item()) <= N

    max_blocks_per_seq = (int(context_lens.max().item()) + block_size - 1) // block_size
    num_blocks_total = B * max_blocks_per_seq  # one physical block per (batch, logical_block)

    K_cache = torch.zeros(
        num_blocks_total, block_size, H, d, dtype=K.dtype, device=device
    )
    V_cache = torch.zeros_like(K_cache)
    block_table = torch.full(
        (B, max_blocks_per_seq), -1, dtype=torch.int32, device=device
    )

    block_counter = 0
    for b in range(B):
        ctx_len = int(context_lens[b].item())
        num_valid_blocks = (ctx_len + block_size - 1) // block_size
        for blk in range(num_valid_blocks):
            tok_start = blk * block_size
            tok_end = min(tok_start + block_size, ctx_len)
            valid = tok_end - tok_start

            # Source slice: K[b, :, tok_start:tok_end, :] — shape (H, valid, d).
            # Destination wants (valid, H, d), so permute.
            k_src = K[b, :, tok_start:tok_end, :].permute(1, 0, 2).contiguous()
            v_src = V[b, :, tok_start:tok_end, :].permute(1, 0, 2).contiguous()

            K_cache[block_counter, :valid] = k_src
            V_cache[block_counter, :valid] = v_src
            block_table[b, blk] = block_counter
            block_counter += 1

    return K_cache, V_cache, block_table, context_lens


def paged_attention_ref(
    Q: torch.Tensor,
    K_cache: torch.Tensor,
    V_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """PyTorch reference for decode-mode paged attention (MHA or GQA).

    For each sequence, gathers all valid K/V tokens from the paged cache
    back into contiguous form, then runs standard scaled dot-product
    attention. O(B * ctx_len) memory allocations — slow but obviously
    correct.

    GQA: if H_q != H_kv, we require H_q % H_kv == 0 and repeat-interleave
    K/V along the head axis so each query head attends to its KV group.

    Args:
        Q: (B, H_q, d). Single query per sequence (decode step).
        K_cache, V_cache: (num_blocks_total, block_size, H_kv, d).
        block_table: (B, max_blocks_per_seq) int32.
        context_lens: (B,) int32.
        scale: softmax scale. Default 1/sqrt(d).

    Returns:
        (B, H_q, d) attention output.
    """
    B, H_q, d = Q.shape
    num_blocks, block_size, H_kv, d_kv = K_cache.shape
    assert H_q % H_kv == 0, f"H_q={H_q} must be divisible by H_kv={H_kv}"
    assert d == d_kv
    gqa_group_size = H_q // H_kv

    if scale is None:
        scale = 1.0 / (d ** 0.5)

    out = torch.zeros_like(Q)

    for b in range(B):
        ctx_len = int(context_lens[b].item())
        if ctx_len == 0:
            continue
        num_valid_blocks = (ctx_len + block_size - 1) // block_size

        # Gather K/V slices for this sequence
        k_pieces = []
        v_pieces = []
        for logical_blk in range(num_valid_blocks):
            phys_blk = int(block_table[b, logical_blk].item())
            assert phys_blk >= 0, f"block_table[{b},{logical_blk}] is -1"
            tok_start = logical_blk * block_size
            tok_end = min(tok_start + block_size, ctx_len)
            valid = tok_end - tok_start
            k_pieces.append(K_cache[phys_blk, :valid])  # (valid, H_kv, d)
            v_pieces.append(V_cache[phys_blk, :valid])

        K_full = torch.cat(k_pieces, dim=0)   # (ctx_len, H_kv, d)
        V_full = torch.cat(v_pieces, dim=0)

        # Per-head scaled dot product; expand KV heads for GQA.
        q = Q[b]                                       # (H_q, d)
        k = K_full.permute(1, 0, 2)                    # (H_kv, ctx_len, d)
        v = V_full.permute(1, 0, 2)                    # (H_kv, ctx_len, d)
        if gqa_group_size != 1:
            k = k.repeat_interleave(gqa_group_size, dim=0)  # (H_q, ctx_len, d)
            v = v.repeat_interleave(gqa_group_size, dim=0)

        scores = torch.einsum("hd,hnd->hn", q, k).float() * scale   # fp32 for stability
        probs = torch.softmax(scores, dim=-1).to(v.dtype)           # (H_q, ctx_len)
        out[b] = torch.einsum("hn,hnd->hd", probs, v)

    return out


def naive_decode_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    context_lens: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Standard (non-paged) decode attention reference. MHA or GQA.

    Used as the "known-good" output that paged_attention_ref must match.

    GQA: if Q has H_q heads and K/V have H_kv heads with H_q > H_kv,
    we repeat-interleave K/V along the head axis by H_q / H_kv.

    Args:
        Q: (B, H_q, d).
        K, V: (B, H_kv, N, d).
        context_lens: (B,) int. If given, mask tokens beyond context_lens[b].
        scale: default 1/sqrt(d).

    Returns:
        (B, H_q, d).
    """
    B_q, H_q, d_q = Q.shape
    B, H_kv, N, d = K.shape
    assert B == B_q and d == d_q
    assert H_q % H_kv == 0, f"H_q={H_q} must be divisible by H_kv={H_kv}"
    gqa_group_size = H_q // H_kv
    if scale is None:
        scale = 1.0 / (d ** 0.5)

    if gqa_group_size != 1:
        K = K.repeat_interleave(gqa_group_size, dim=1)   # (B, H_q, N, d)
        V = V.repeat_interleave(gqa_group_size, dim=1)

    # scores: (B, H_q, N) = einsum over d
    scores = torch.einsum("bhd,bhnd->bhn", Q, K).float() * scale

    if context_lens is not None:
        token_idx = torch.arange(N, device=Q.device)               # (N,)
        # mask[b, n] = n < context_lens[b]
        mask = token_idx[None, :] < context_lens.to(Q.device)[:, None]   # (B, N)
        scores = scores.masked_fill(~mask[:, None, :], float("-inf"))

    probs = torch.softmax(scores, dim=-1).to(V.dtype)   # (B, H_q, N)
    return torch.einsum("bhn,bhnd->bhd", probs, V)
