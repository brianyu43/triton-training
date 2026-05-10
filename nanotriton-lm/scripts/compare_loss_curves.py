from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_metrics(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if len(rows) < 2:
        raise ValueError(f"{path} must contain at least two metric rows")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and Triton loss curves.")
    parser.add_argument("--ref", required=True, type=Path)
    parser.add_argument("--triton", required=True, type=Path)
    parser.add_argument("--max-final-val-diff", type=float, default=0.25)
    return parser.parse_args()


def summarize(name: str, rows: list[dict]) -> dict:
    first = rows[0]
    final = rows[-1]
    return {
        "name": name,
        "first_iter": first["iter"],
        "final_iter": final["iter"],
        "first_train": first["train"],
        "final_train": final["train"],
        "first_val": first["val"],
        "final_val": final["val"],
        "train_delta": final["train"] - first["train"],
        "val_delta": final["val"] - first["val"],
    }


def main() -> None:
    args = parse_args()
    ref = load_metrics(args.ref)
    triton = load_metrics(args.triton)
    ref_summary = summarize("ref", ref)
    triton_summary = summarize("triton", triton)
    final_val_diff = abs(ref_summary["final_val"] - triton_summary["final_val"])
    result = {
        "ref": ref_summary,
        "triton": triton_summary,
        "final_val_abs_diff": final_val_diff,
        "max_final_val_diff": args.max_final_val_diff,
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    failures = []
    if ref_summary["val_delta"] >= 0:
        failures.append("reference validation loss did not decrease")
    if triton_summary["val_delta"] >= 0:
        failures.append("Triton validation loss did not decrease")
    if final_val_diff > args.max_final_val_diff:
        failures.append(
            f"final validation loss diff {final_val_diff:.4f} exceeds {args.max_final_val_diff:.4f}"
        )
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
