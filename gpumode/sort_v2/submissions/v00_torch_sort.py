#!POPCORN leaderboard sort_v2
#!POPCORN gpus A100

import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    values, output = data
    output[...] = torch.sort(values)[0]
    return output
