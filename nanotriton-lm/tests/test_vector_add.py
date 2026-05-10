from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for Triton kernel tests", allow_module_level=True)

from nanotriton.kernels.vector_add import vector_add


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(1024,), (7777,), (2, 17, 513)])
def test_vector_add_matches_torch(dtype, shape) -> None:
    torch.manual_seed(1337)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    y = torch.randn(shape, device="cuda", dtype=dtype)
    actual = vector_add(x, y)
    expected = x + y
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_vector_add_accepts_non_contiguous_input() -> None:
    x_base = torch.randn((8, 16), device="cuda", dtype=torch.float32)
    y_base = torch.randn((8, 16), device="cuda", dtype=torch.float32)
    x = x_base[:, ::2]
    y = y_base[:, ::2]
    actual = vector_add(x, y)
    torch.testing.assert_close(actual, x + y, rtol=0.0, atol=0.0)
