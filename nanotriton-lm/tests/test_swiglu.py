from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for Triton kernel tests", allow_module_level=True)

from nanotriton.autograd.swiglu_fn import triton_swiglu
from nanotriton.kernels.swiglu import swiglu_backward, swiglu_forward


def swiglu_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(a.float()).to(dtype=a.dtype) * b


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(1024,), (7777,), (2, 17, 513), (16, 128, 512)])
def test_swiglu_forward_matches_torch(dtype, shape) -> None:
    torch.manual_seed(1337)
    a = torch.randn(shape, device="cuda", dtype=dtype)
    b = torch.randn(shape, device="cuda", dtype=dtype)
    actual = swiglu_forward(a, b)
    expected = swiglu_reference(a, b)
    atol = 1e-5 if dtype is torch.float32 else 2e-3
    rtol = 1e-5 if dtype is torch.float32 else 2e-3
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(1024,), (7777,), (2, 17, 513), (16, 128, 512)])
def test_swiglu_backward_matches_torch_autograd(dtype, shape) -> None:
    torch.manual_seed(1337)
    a = torch.randn(shape, device="cuda", dtype=dtype)
    b = torch.randn(shape, device="cuda", dtype=dtype)
    grad_out = torch.randn(shape, device="cuda", dtype=dtype)

    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    expected = swiglu_reference(a_ref, b_ref)
    expected.backward(grad_out)

    actual_da, actual_db = swiglu_backward(grad_out, a, b)
    atol = 1e-5 if dtype is torch.float32 else 3e-3
    rtol = 1e-5 if dtype is torch.float32 else 3e-3
    torch.testing.assert_close(actual_da, a_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_db, b_ref.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_triton_swiglu_autograd_function_matches_torch(dtype) -> None:
    torch.manual_seed(1337)
    shape = (4, 9, 32)
    a = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    b = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad_out = torch.randn(shape, device="cuda", dtype=dtype)

    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    actual = triton_swiglu(a, b)
    expected = swiglu_reference(a_ref, b_ref)
    actual.backward(grad_out)
    expected.backward(grad_out)

    atol = 1e-5 if dtype is torch.float32 else 3e-3
    rtol = 1e-5 if dtype is torch.float32 else 3e-3
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(a.grad, a_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(b.grad, b_ref.grad, atol=atol, rtol=rtol)


def test_swiglu_accepts_non_contiguous_input() -> None:
    torch.manual_seed(1337)
    a_base = torch.randn((4, 8, 32), device="cuda", dtype=torch.float32)
    b_base = torch.randn((4, 8, 32), device="cuda", dtype=torch.float32)
    a = a_base[:, :, ::2]
    b = b_base[:, :, ::2]
    actual = swiglu_forward(a, b)
    expected = swiglu_reference(a, b)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
