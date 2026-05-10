from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nanotriton.tokenizer import CharTokenizer

DEFAULT_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Tiny Shakespeare char-level dataset.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out-dir", default=Path(__file__).resolve().parent)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = out_dir / "input.txt"
    if not input_path.exists():
        print(f"downloading {args.url}")
        urllib.request.urlretrieve(args.url, input_path)

    text = input_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    data = np.array(tokenizer.encode(text), dtype=np.uint16)
    split_idx = int(len(data) * (1.0 - args.val_ratio))
    train_ids = data[:split_idx]
    val_ids = data[split_idx:]
    train_ids.tofile(out_dir / "train.bin")
    val_ids.tofile(out_dir / "val.bin")
    tokenizer.save(out_dir / "meta.json")
    print(f"chars: {len(text)}")
    print(f"vocab_size: {tokenizer.vocab_size}")
    print(f"train tokens: {len(train_ids)}")
    print(f"val tokens: {len(val_ids)}")


if __name__ == "__main__":
    main()
