from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for Triton kernel tests", allow_module_level=True)

from nanotriton.autograd.rmsnorm_fn import triton_rmsnorm
from nanotriton.kernels.rmsnorm import rmsnorm_backward, rmsnorm_forward
from nanotriton.modules.norm import TritonRMSNorm


def rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    return (x_float * torch.rsqrt(variance + eps) * weight.float()).to(dtype=x.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(2, 8, 16), (3, 17, 65), (16, 128, 128)])
def test_rmsnorm_forward_matches_torch(dtype, shape) -> None:
    torch.manual_seed(1337)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn((shape[-1],), device="cuda", dtype=dtype)
    actual = rmsnorm_forward(x, weight, eps=1e-6)
    expected = rmsnorm_reference(x, weight, eps=1e-6)
    atol = 1e-5 if dtype is torch.float32 else 2e-3
    rtol = 1e-5 if dtype is torch.float32 else 2e-3
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def test_rmsnorm_forward_accepts_non_contiguous_input() -> None:
    torch.manual_seed(1337)
    x_base = torch.randn((4, 8, 32), device="cuda", dtype=torch.float32)
    x = x_base[:, :, ::2]
    weight = torch.randn((x.shape[-1],), device="cuda", dtype=torch.float32)
    actual = rmsnorm_forward(x, weight)
    expected = rmsnorm_reference(x, weight, eps=1e-6)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(2, 8, 16), (3, 17, 65), (16, 128, 128)])
def test_rmsnorm_backward_matches_torch_autograd(dtype, shape) -> None:
    torch.manual_seed(1337)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn((shape[-1],), device="cuda", dtype=dtype)
    grad_out = torch.randn(shape, device="cuda", dtype=dtype)

    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    expected_y = rmsnorm_reference(x_ref, weight_ref, eps=1e-6)
    expected_y.backward(grad_out)

    actual_dx, actual_dweight = rmsnorm_backward(grad_out, x, weight, eps=1e-6)
    atol = 1e-5 if dtype is torch.float32 else 3e-3
    rtol = 1e-5 if dtype is torch.float32 else 3e-3
    torch.testing.assert_close(actual_dx, x_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_dweight, weight_ref.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_triton_rmsnorm_autograd_function_matches_torch(dtype) -> None:
    torch.manual_seed(1337)
    shape = (4, 9, 32)
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn((shape[-1],), device="cuda", dtype=dtype, requires_grad=True)
    grad_out = torch.randn(shape, device="cuda", dtype=dtype)

    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)

    actual = triton_rmsnorm(x, weight)
    expected = rmsnorm_reference(x_ref, weight_ref, eps=1e-6)
    actual.backward(grad_out)
    expected.backward(grad_out)

    atol = 1e-5 if dtype is torch.float32 else 3e-3
    rtol = 1e-5 if dtype is torch.float32 else 3e-3
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(weight.grad, weight_ref.grad, atol=atol, rtol=rtol)


def test_triton_rmsnorm_module_matches_reference_module() -> None:
    torch.manual_seed(1337)
    shape = (2, 8, 16)
    module = TritonRMSNorm(shape[-1]).cuda()
    x = torch.randn(shape, device="cuda", dtype=torch.float32, requires_grad=True)
    grad_out = torch.randn(shape, device="cuda", dtype=torch.float32)

    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = module.weight.detach().clone().requires_grad_(True)
    expected = rmsnorm_reference(x_ref, weight_ref, eps=module.eps)
    actual = module(x)
    expected.backward(grad_out)
    actual.backward(grad_out)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(module.weight.grad, weight_ref.grad, atol=1e-5, rtol=1e-5)
