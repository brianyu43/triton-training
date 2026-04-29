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
