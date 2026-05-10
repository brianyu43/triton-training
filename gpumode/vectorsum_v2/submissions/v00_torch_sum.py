#!POPCORN leaderboard vectorsum_py
#!POPCORN gpus A100

import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    values, _output = data
    return values.to(torch.float64).sum().to(torch.float32)
