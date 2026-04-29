from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CharTokenizer:
    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {ch: idx for idx, ch in enumerate(chars)}
        itos = {idx: ch for ch, idx in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[int(idx)] for idx in ids)

    def to_meta(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
        }

    @classmethod
    def from_meta(cls, meta: dict) -> "CharTokenizer":
        stoi = {str(k): int(v) for k, v in meta["stoi"].items()}
        itos = {int(k): str(v) for k, v in meta["itos"].items()}
        return cls(stoi=stoi, itos=itos)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_meta(), handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_meta(json.load(handle))
