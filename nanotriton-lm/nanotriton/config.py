from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 0
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = False
    norm_type: str = "rmsnorm"
    norm_impl: str = "torch"
    mlp_type: str = "swiglu"


@dataclass
class DataConfig:
    dataset: str = "shakespeare_char"
    data_dir: str = "data/shakespeare_char"


@dataclass
class TrainConfig:
    seed: int = 1337
    device: str = "auto"
    dtype: str = "float16"
    compile: bool = False
    out_dir: str = "out/tiny_shakespeare_ref"
    eval_interval: int = 100
    log_interval: int = 10
    eval_iters: int = 20
    max_iters: int = 1000
    batch_size: int = 16
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_iters: int = 50
    lr_decay_iters: int = 1000
    min_lr: float = 6e-5
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def _build_dataclass(cls: type, payload: dict[str, Any]):
    return cls(**payload)


def load_config(path: str | Path) -> TrainConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw["data"] = _build_dataclass(DataConfig, raw.get("data", {}))
    raw["model"] = _build_dataclass(ModelConfig, raw.get("model", {}))
    return _build_dataclass(TrainConfig, raw)


def config_to_dict(config: TrainConfig | ModelConfig | DataConfig) -> dict[str, Any]:
    return asdict(config)


def model_config_from_dict(payload: dict[str, Any]) -> ModelConfig:
    return _build_dataclass(ModelConfig, payload)
