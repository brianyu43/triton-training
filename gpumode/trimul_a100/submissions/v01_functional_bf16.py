import torch
import torch.nn.functional as F
from task import input_t, output_t


@torch.no_grad()
def fallback_functional(x, mask, weights, dim: int, hidden_dim: int):
    x = F.layer_norm(
        x,
        (dim,),
        weights["norm.weight"],
        weights["norm.bias"],
    )

    left = F.linear(x, weights["left_proj.weight"])
    right = F.linear(x, weights["right_proj.weight"])

    mask_expanded = mask.unsqueeze(-1)
    left = left * mask_expanded
    right = right * mask_expanded

    left_gate = torch.sigmoid(F.linear(x, weights["left_gate.weight"]))
    right_gate = torch.sigmoid(F.linear(x, weights["right_gate.weight"]))
    out_gate = torch.sigmoid(F.linear(x, weights["out_gate.weight"]))

    left = left * left_gate
    right = right * right_gate

    out = torch.einsum(
        "... i k d, ... j k d -> ... i j d",
        left.to(torch.bfloat16),
        right.to(torch.bfloat16),
    ).to(torch.float32)

    out = F.layer_norm(
        out,
        (hidden_dim,),
        weights["to_out_norm.weight"],
        weights["to_out_norm.bias"],
    )
    out = out * out_gate
    return F.linear(out, weights["to_out.weight"]).to(torch.float32)


@torch.no_grad()
def custom_kernel(data: input_t) -> output_t:
    input_tensor, mask, weights, config = data
    return fallback_functional(
        input_tensor,
        mask,
        weights,
        dim=config["dim"],
        hidden_dim=config["hidden_dim"],
    )
