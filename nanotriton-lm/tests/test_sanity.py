from __future__ import annotations

from nanotriton.config import load_config
from nanotriton.tokenizer import CharTokenizer


def test_char_tokenizer_roundtrip() -> None:
    tokenizer = CharTokenizer.from_text("abba\n")
    encoded = tokenizer.encode("baba")
    assert tokenizer.decode(encoded) == "baba"
    assert tokenizer.vocab_size == 3


def test_load_config() -> None:
    config = load_config("configs/tiny_shakespeare_ref.yaml")
    assert config.model.block_size == 128
    assert config.model.norm_type == "rmsnorm"
    assert config.model.mlp_type == "swiglu"
