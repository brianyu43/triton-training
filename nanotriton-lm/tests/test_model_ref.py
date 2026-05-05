from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")


def test_gpt_forward_and_generate() -> None:
    import torch

    from nanotriton.config import ModelConfig
    from nanotriton.model_ref import GPT

    config = ModelConfig(vocab_size=16, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (2, config.block_size))
    logits, loss = model(idx, idx)
    assert logits.shape == (2, config.block_size, config.vocab_size)
    assert loss is not None
    generated = model.generate(idx[:, :2], max_new_tokens=3, top_k=4)
    assert generated.shape == (2, 5)


@pytest.mark.skipif(
    importlib.util.find_spec("triton") is None,
    reason="triton is not installed",
)
@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or not __import__("torch").cuda.is_available(),
    reason="CUDA is required for Triton model test",
)
def test_gpt_with_triton_rmsnorm_forward_backward() -> None:
    import torch

    from nanotriton.config import ModelConfig
    from nanotriton.model_ref import GPT
    from nanotriton.modules.norm import TritonRMSNorm

    config = ModelConfig(
        vocab_size=16,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        norm_impl="triton",
    )
    model = GPT(config).cuda()
    assert isinstance(model.transformer["ln_f"], TritonRMSNorm)
    idx = torch.randint(0, config.vocab_size, (2, config.block_size), device="cuda")
    logits, loss = model(idx, idx)
    assert logits.shape == (2, config.block_size, config.vocab_size)
    assert loss is not None
    loss.backward()
    assert model.transformer["ln_f"].weight.grad is not None


@pytest.mark.skipif(
    importlib.util.find_spec("triton") is None,
    reason="triton is not installed",
)
@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or not __import__("torch").cuda.is_available(),
    reason="CUDA is required for Triton model test",
)
def test_gpt_with_triton_swiglu_forward_backward() -> None:
    import torch

    from nanotriton.config import ModelConfig
    from nanotriton.model_ref import GPT

    config = ModelConfig(
        vocab_size=16,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        mlp_impl="triton_swiglu",
    )
    model = GPT(config).cuda()
    assert model.transformer["h"][0].mlp.mlp_impl == "triton_swiglu"
    idx = torch.randint(0, config.vocab_size, (2, config.block_size), device="cuda")
    logits, loss = model(idx, idx)
    assert logits.shape == (2, config.block_size, config.vocab_size)
    assert loss is not None
    loss.backward()
    assert model.transformer["h"][0].mlp.w1.weight.grad is not None
    assert model.transformer["h"][0].mlp.w3.weight.grad is not None
