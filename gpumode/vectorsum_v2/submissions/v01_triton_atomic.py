#!POPCORN leaderboard vectorsum_py
#!POPCORN gpus A100

import torch
import triton
import triton.language as tl
from task import input_t, output_t


@triton.jit
def _sum_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    block_sum = tl.sum(x, axis=0)
    tl.atomic_add(output_ptr, block_sum)


def custom_kernel(data: input_t) -> output_t:
    values, output = data
    n_elements = values.numel()

    if n_elements <= 4096:
        return values.to(torch.float64).sum().to(torch.float32)

    output.zero_()
    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    _sum_kernel[grid](values, output, n_elements, BLOCK_SIZE=block_size)
    return output[0]
