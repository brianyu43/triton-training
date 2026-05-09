#!POPCORN leaderboard matmul_v2
#!POPCORN gpus A100

import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    a, b, c = data
    return torch.ops.aten.mm.out(a, b, out=c)
