from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from nanotriton.checkpoint import save_checkpoint
from nanotriton.config import config_to_dict, load_config
from nanotriton.model_ref import GPT
from nanotriton.tokenizer import CharTokenizer
from nanotriton.utils import (
    append_jsonl,
    get_device,
    learning_rate_for_iter,
    resolve_project_path,
    set_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PyTorch reference GPT.")
    parser.add_argument("--config", default="configs/tiny_shakespeare_ref.yaml")
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def torch_dtype(name: str):
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def load_split(data_dir: Path, split: str) -> np.memmap:
    return np.memmap(data_dir / f"{split}.bin", dtype=np.uint16, mode="r")


def get_batch(split_data: np.memmap, block_size: int, batch_size: int, device: str):
    import torch

    max_start = len(split_data) - block_size - 1
    if max_start <= 0:
        raise ValueError("Dataset split is too small for the configured block_size")
    ix = torch.randint(max_start, (batch_size,))
    x = torch.stack(
        [
            torch.from_numpy(np.array(split_data[int(i) : int(i) + block_size], dtype=np.int64, copy=True))
            for i in ix
        ]
    )
    y = torch.stack(
        [
            torch.from_numpy(np.array(split_data[int(i) + 1 : int(i) + 1 + block_size], dtype=np.int64, copy=True))
            for i in ix
        ]
    )
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def estimate_loss(model, train_data, val_data, config, device: str, ctx):
    import torch

    model.eval()
    out = {}
    with torch.no_grad():
        for split, data in (("train", train_data), ("val", val_data)):
            losses = torch.zeros(config.eval_iters, device=device)
            for idx in range(config.eval_iters):
                x, y = get_batch(data, config.model.block_size, config.batch_size, device)
                with ctx:
                    _, loss = model(x, y)
                losses[idx] = loss
            out[split] = losses.mean().item()
    model.train()
    return out


def main() -> None:
    import torch

    args = parse_args()
    config = load_config(resolve_project_path(args.config))
    if args.max_iters is not None:
        config.max_iters = args.max_iters
    if args.device is not None:
        config.device = args.device
    if args.out_dir is not None:
        config.out_dir = args.out_dir

    set_seed(config.seed)
    device = get_device(config.device)
    device_type = "cuda" if device.startswith("cuda") else device
    dtype = torch_dtype(config.dtype)
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=dtype)
    )

    data_dir = resolve_project_path(config.data.data_dir)
    tokenizer = CharTokenizer.load(data_dir / "meta.json")
    config.model.vocab_size = tokenizer.vocab_size
    train_data = load_split(data_dir, "train")
    val_data = load_split(data_dir, "val")

    out_dir = resolve_project_path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "config.json", config_to_dict(config))

    model = GPT(config.model).to(device)
    if config.compile:
        model = torch.compile(model)
    optimizer = model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )

    best_val_loss = float("inf")
    start_time = time.time()
    for iter_num in range(config.max_iters + 1):
        lr = learning_rate_for_iter(
            iter_num,
            config.learning_rate,
            config.warmup_iters,
            config.lr_decay_iters,
            config.min_lr,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % config.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, config, device, ctx)
            elapsed = time.time() - start_time
            metrics = {"iter": iter_num, "lr": lr, "elapsed_s": elapsed, **losses}
            append_jsonl(out_dir / "metrics.jsonl", metrics)
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}, lr {lr:.2e}"
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(
                    out_dir / "checkpoint.pt",
                    model=model,
                    optimizer=optimizer,
                    model_config=config.model,
                    iteration=iter_num,
                    best_val_loss=best_val_loss,
                    extra={"config": config_to_dict(config)},
                )

        if iter_num == config.max_iters:
            break

        x, y = get_batch(train_data, config.model.block_size, config.batch_size, device)
        with ctx:
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()


if __name__ == "__main__":
    main()
