#!POPCORN leaderboard matmul_v2
#!POPCORN gpus A100

import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    a, b, c = data
    return torch.addmm(c, a, b, beta=0.0, alpha=1.0, out=c)
