from __future__ import annotations

from pathlib import Path
from typing import Any

from nanotriton.config import ModelConfig, model_config_from_dict


def save_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer,
    model_config: ModelConfig,
    iteration: int,
    best_val_loss: float,
    extra: dict[str, Any] | None = None,
) -> None:
    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "model_config": model.config_dict() if hasattr(model, "config_dict") else model_config.__dict__,
        "iter_num": iteration,
        "best_val_loss": best_val_loss,
        "extra": extra or {},
    }
    torch.save(payload, target)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    import torch

    return torch.load(Path(path), map_location=map_location)


def load_model_config(path: str | Path) -> ModelConfig:
    checkpoint = load_checkpoint(path, map_location="cpu")
    return model_config_from_dict(checkpoint["model_config"])
