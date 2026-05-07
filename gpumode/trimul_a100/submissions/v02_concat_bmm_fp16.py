import torch
import torch.nn.functional as F
from task import input_t, output_t


@torch.no_grad()
def concat_bmm_path(x, mask, weights, dim: int, hidden_dim: int):
    batch, seq_len, _, _ = x.shape
    rows = batch * seq_len * seq_len

    x_norm = F.layer_norm(
        x,
        (dim,),
        weights["norm.weight"],
        weights["norm.bias"],
    )
    x_flat = x_norm.reshape(rows, dim).to(torch.float16)

    # Weight order matches split below. This allocation is measured on purpose:
    # it tells us whether one large GEMM beats five smaller launches.
    w5 = torch.cat(
        [
            weights["left_proj.weight"],
            weights["right_proj.weight"],
            weights["left_gate.weight"],
            weights["right_gate.weight"],
            weights["out_gate.weight"],
        ],
        dim=0,
    ).t().contiguous().to(torch.float16)

    all_proj = x_flat @ w5
    left, right, left_gate, right_gate, out_gate = all_proj.split(hidden_dim, dim=1)

    mask_flat = mask.reshape(rows, 1).to(torch.float16)
    left = left * torch.sigmoid(left_gate) * mask_flat
    right = right * torch.sigmoid(right_gate) * mask_flat
    out_gate = torch.sigmoid(out_gate)

    left = left.view(batch, seq_len, seq_len, hidden_dim).permute(0, 3, 1, 2)
    right = right.view(batch, seq_len, seq_len, hidden_dim).permute(0, 3, 1, 2)
    hidden = torch.bmm(
        left.reshape(batch * hidden_dim, seq_len, seq_len),
        right.reshape(batch * hidden_dim, seq_len, seq_len).transpose(1, 2),
    )

    hidden_flat = hidden.view(batch, hidden_dim, seq_len, seq_len).permute(0, 2, 3, 1)
    hidden_flat = hidden_flat.reshape(rows, hidden_dim).to(torch.float32)
    hidden_flat = F.layer_norm(
        hidden_flat,
        (hidden_dim,),
        weights["to_out_norm.weight"],
        weights["to_out_norm.bias"],
    ).to(torch.float16)

    gated = hidden_flat * out_gate
    out = gated @ weights["to_out.weight"].t().to(torch.float16)
    return out.view(batch, seq_len, seq_len, dim).to(torch.float32)


@torch.no_grad()
def custom_kernel(data: input_t) -> output_t:
    input_tensor, mask, weights, config = data
    return concat_bmm_path(
        input_tensor,
        mask,
        weights,
        dim=config["dim"],
        hidden_dim=config["hidden_dim"],
    )
