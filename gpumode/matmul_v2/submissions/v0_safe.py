#!POPCORN leaderboard matmul_v2
#!POPCORN gpus L4

import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    a, b, c = data
    torch.mm(a, b, out=c)
    return c
