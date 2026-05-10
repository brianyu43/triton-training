"""Correctness tests for the custom ops.

Verifies that both `torch.ops.mylib.flash_attention` and
`torch.ops.mylib.naive_attention` match PyTorch's built-in
`F.scaled_dot_product_attention` within FP32 tolerance.

Also demonstrates the contiguous / dtype / device guard behavior.
"""

from __future__ import annotations

import sys
import torch
import torch.nn.functional as F

# Import the built extension. This registers torch.ops.mylib.*.
import mylib_ext  # noqa: F401


def sdpa_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Reference attention via PyTorch SDPA. Expects (N, d), returns (N, d)."""
    # SDPA expects (batch, heads, seq, dim). Unsqueeze + re-squeeze.
    q4 = q.unsqueeze(0).unsqueeze(0)
    k4 = k.unsqueeze(0).unsqueeze(0)
    v4 = v.unsqueeze(0).unsqueeze(0)
    out = F.scaled_dot_product_attention(q4, k4, v4)
    return out.squeeze(0).squeeze(0)


def run_correctness(name: str, op, q, k, v, tol_abs=1e-4, tol_rel=1e-3):
    out = op(q, k, v)
    ref = sdpa_reference(q, k, v)
    abs_err = (out - ref).abs().max().item()
    rel_err = ((out - ref).abs() / (ref.abs() + 1e-6)).max().item()
    ok = abs_err < tol_abs
    status = "OK " if ok else "FAIL"
    print(f"  [{status}] {name:8s}  max_abs={abs_err:.3e}  max_rel={rel_err:.3e}")
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available — extension cannot run.")
        return 1

    device = torch.device("cuda")
    torch.manual_seed(42)

    passes = 0
    total = 0
    for N in (128, 512, 1024, 2048):
        print(f"\nN = {N}")
        q = torch.randn(N, 64, device=device, dtype=torch.float32)
        k = torch.randn(N, 64, device=device, dtype=torch.float32)
        v = torch.randn(N, 64, device=device, dtype=torch.float32)

        if N <= 2048:  # naive softmax smem fits
            total += 1
            if run_correctness("naive", torch.ops.mylib.naive_attention, q, k, v):
                passes += 1
        total += 1
        if run_correctness("flash", torch.ops.mylib.flash_attention, q, k, v):
            passes += 1

    # Guard demonstrations -----------------------------------------------------
    print("\n[guard] non-contiguous input should raise:")
    N = 256
    big = torch.randn(N, 64 * 2, device=device, dtype=torch.float32)
    q_noncontig = big[:, ::2]          # stride (128, 2) — not contiguous
    assert not q_noncontig.is_contiguous()
    try:
        torch.ops.mylib.flash_attention(q_noncontig, q_noncontig, q_noncontig)
        print("  [FAIL] did not raise")
    except RuntimeError as e:
        msg = str(e).splitlines()[0]
        print(f"  [OK  ] raised: {msg}")

    print("\n[guard] wrong dtype (fp16) should raise:")
    q_half = torch.randn(N, 64, device=device, dtype=torch.float16)
    try:
        torch.ops.mylib.flash_attention(q_half, q_half, q_half)
        print("  [FAIL] did not raise")
    except RuntimeError as e:
        msg = str(e).splitlines()[0]
        print(f"  [OK  ] raised: {msg}")

    print("\n[guard] CPU tensor should raise:")
    q_cpu = torch.randn(N, 64, dtype=torch.float32)
    try:
        torch.ops.mylib.flash_attention(q_cpu, q_cpu, q_cpu)
        print("  [FAIL] did not raise")
    except RuntimeError as e:
        msg = str(e).splitlines()[0]
        print(f"  [OK  ] raised: {msg}")

    print(f"\n{passes}/{total} correctness checks passed")
    return 0 if passes == total else 1


if __name__ == "__main__":
    sys.exit(main())
