#!POPCORN leaderboard vectoradd_v2
#!POPCORN gpus L4

from task import input_t, output_t

import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets = tl.max_contiguous(tl.multiple_of(offsets, BLOCK_SIZE), BLOCK_SIZE)

    a = tl.load(a_ptr + offsets, eviction_policy="evict_first")
    b = tl.load(b_ptr + offsets, eviction_policy="evict_first")
    tl.store(out_ptr + offsets, a + b)


@triton.jit
def _add_kernel_masked(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, a + b, mask=mask)


def custom_kernel(data: input_t) -> output_t:
    # vector-add-v2 passes an output buffer in the input tuple.  Reusing it
    # avoids a very visible allocation cost on the 8192/16384 cases.
    if len(data) == 3:
        A, B, output = data
    else:
        A, B = data
        output = torch.empty_like(A)

    n_elements = A.numel()
    block_size = 512
    grid = (triton.cdiv(n_elements, block_size),)
    if n_elements % block_size == 0:
        _add_kernel[grid](A, B, output, BLOCK_SIZE=block_size, num_warps=8)
    else:
        _add_kernel_masked[grid](
            A, B, output, n_elements, BLOCK_SIZE=block_size, num_warps=8
        )
    return output
