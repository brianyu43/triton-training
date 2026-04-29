from __future__ import annotations

import argparse

from nanotriton.checkpoint import load_checkpoint
from nanotriton.config import model_config_from_dict
from nanotriton.model_ref import GPT
from nanotriton.tokenizer import CharTokenizer
from nanotriton.utils import get_device, resolve_project_path, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a reference GPT checkpoint.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--prompt", default="To be")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--meta", default="data/shakespeare_char/meta.json")
    return parser.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    checkpoint = load_checkpoint(resolve_project_path(args.ckpt), map_location=device)
    model_config = model_config_from_dict(checkpoint["model_config"])
    model = GPT(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokenizer = CharTokenizer.load(resolve_project_path(args.meta))
    start_ids = tokenizer.encode(args.prompt)
    idx = torch.tensor([start_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
