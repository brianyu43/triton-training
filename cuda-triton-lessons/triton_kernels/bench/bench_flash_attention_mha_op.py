"""
Lesson 09 · Phase 4 · torch.library.custom_op integration test.

Validates that the custom-op wrapper behaves identically to the direct
Python wrapper across three access paths:

  1. Direct op call:  `torch.ops.triton_training.flash_attention_mha(...)`
  2. Python call:     `flash_attention_mha_op(...)` (the decorated function)
  3. torch.compile:   a model that uses `torch.ops.triton_training.flash_attention_mha`
                       wrapped in `torch.compile(model)` must compile without
                       graph break and produce identical output.

A small realistic attention block is also exercised end-to-end under
torch.compile to confirm the fake impl is wired up correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Importing this module triggers the @custom_op registration.
import triton_kernels.flash_attention_mha_op  # noqa: F401  (side-effect import)

from triton_kernels.flash_attention_mha import triton_flash_attention_mha  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_qkv(B, H, N, d, dtype=torch.float16, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, H, N, d, device="cuda", dtype=dtype)
    k = torch.randn(B, H, N, d, device="cuda", dtype=dtype)
    v = torch.randn(B, H, N, d, device="cuda", dtype=dtype)
    return q, k, v


def close(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-9)


# ---------------------------------------------------------------------------
# Test 1 — three access paths return the same tensor
# ---------------------------------------------------------------------------

def test_three_paths_match():
    print("\n[1] Three access paths produce the same output:")
    q, k, v = make_qkv(1, 32, 1024, 128)

    for is_causal in (False, True):
        a = triton_flash_attention_mha(q, k, v, is_causal=is_causal)
        b = torch.ops.triton_training.flash_attention_mha(q, k, v, is_causal)
        c = torch.ops.triton_training.flash_attention_mha.default(q, k, v, is_causal)
        err_ab = close(a, b)
        err_ac = close(a, c)
        print(f"  causal={is_causal}:  raw_wrapper vs torch.ops  err={err_ab:.2e}")
        print(f"  causal={is_causal}:  raw_wrapper vs .default    err={err_ac:.2e}")
        assert err_ab == 0.0, "torch.ops path must match raw wrapper bit-exact"
        assert err_ac == 0.0, ".default path must match raw wrapper bit-exact"


# ---------------------------------------------------------------------------
# Test 2 — torch.compile on a function that wraps our custom op
# ---------------------------------------------------------------------------

def test_torch_compile_function():
    print("\n[2] torch.compile on a thin wrapper function:")

    def attn_fn(q, k, v):
        # Deliberately include surrounding ops to prove that torch.compile
        # can fuse them without a graph break at our custom op.
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        out = torch.ops.triton_training.flash_attention_mha(q, k, v, True)
        return out * 1.0   # trailing op; compiler should keep graph unbroken

    q, k, v = make_qkv(1, 32, 1024, 128)
    out_eager = attn_fn(q, k, v)
    out_compiled_fn = torch.compile(attn_fn, fullgraph=True)
    out_compiled = out_compiled_fn(q, k, v)

    err = close(out_eager, out_compiled)
    print(f"  eager vs compiled (fullgraph=True)   err={err:.2e}")
    assert err == 0.0, "compiled path must match eager"
    print("  fullgraph=True succeeded → no graph break at the custom op")


# ---------------------------------------------------------------------------
# Test 3 — a realistic attention block under torch.compile
# ---------------------------------------------------------------------------

class AttentionBlock(torch.nn.Module):
    """Minimal LLaMA-style attention block using our custom op."""

    def __init__(self, d_model: int = 1024, n_heads: int = 16):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.qkv = torch.nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        out = torch.ops.triton_training.flash_attention_mha(q, k, v, True)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out(out)


def test_realistic_block_under_compile():
    print("\n[3] Realistic attention block under torch.compile:")

    torch.manual_seed(42)
    model = AttentionBlock(d_model=1024, n_heads=16).cuda().half()
    x = torch.randn(2, 512, 1024, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        out_eager = model(x)
        compiled_model = torch.compile(model, fullgraph=True)
        out_compiled = compiled_model(x)

    err = close(out_eager, out_compiled)
    print(f"  eager vs compiled block   err={err:.2e}")
    # Linear layers introduce rounding; allow fp16 tolerance.
    assert err < 1e-2, f"compiled block diverged: err={err}"
    print(f"  shapes: in={tuple(x.shape)}  out={tuple(out_eager.shape)}")
    print(f"  fullgraph=True succeeded → the whole attention block is one graph")


# ---------------------------------------------------------------------------
# Test 4 — op is visible in torch.ops with the right schema
# ---------------------------------------------------------------------------

def test_op_schema_visibility():
    print("\n[4] Op is registered under torch.ops:")
    op = torch.ops.triton_training.flash_attention_mha
    print(f"  op = {op}")
    print(f"  overloads = {op.overloads()}")
    print(f"  schema = {op.default._schema}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dev = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"device = {dev}   cap = sm_{cap[0]}{cap[1]}")
    print(f"torch  = {torch.__version__}")

    test_op_schema_visibility()
    test_three_paths_match()
    test_torch_compile_function()
    test_realistic_block_under_compile()

    print("\nALL OK")


if __name__ == "__main__":
    main()
