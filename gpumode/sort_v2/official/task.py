from typing import TypedDict, TypeAlias
import torch

input_t: TypeAlias = tuple[torch.Tensor, torch.Tensor]
output_t: TypeAlias = torch.Tensor

class TestSpec(TypedDict):
    size: int
    seed: int 
