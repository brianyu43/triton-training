#!/usr/bin/env python3
import argparse
import csv
import math
import re
from pathlib import Path


SPEC_RE = re.compile(r"([a-zA-Z]+):([^;]+)")


def parse_log(path: Path):
    values = {}
    for line in path.read_text(errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()

    rows = []
    count = int(values.get("benchmark-count", "0") or "0")
    for idx in range(count):
        prefix = f"benchmark.{idx}."
        spec = values.get(prefix + "spec", "")
        if not spec:
            continue
        fields = {k: v for k, v in SPEC_RE.findall(spec)}
        row = {
            "log": str(path),
            "idx": idx,
            "bs": fields.get("bs", ""),
            "N": fields.get("seqlen", ""),
            "C": fields.get("dim", ""),
            "H": fields.get("hiddendim", ""),
            "nomask": fields.get("nomask", ""),
            "distribution": fields.get("distribution", ""),
            "runs": values.get(prefix + "runs", ""),
            "mean_us": ns_to_us(values.get(prefix + "mean", "")),
            "best_us": ns_to_us(values.get(prefix + "best", "")),
            "std_us": ns_to_us(values.get(prefix + "std", "")),
            "err_us": ns_to_us(values.get(prefix + "err", "")),
        }
        rows.append(row)
    return rows


def ns_to_us(value: str):
    if not value:
        return ""
    return f"{float(value) / 1000.0:.3f}"


def geomean_us(rows):
    means = [float(r["mean_us"]) for r in rows if r.get("mean_us")]
    if not means:
        return None
    return math.exp(sum(math.log(v) for v in means) / len(means))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--csv", type=Path, help="Append parsed rows to this CSV")
    args = parser.parse_args()

    all_rows = []
    for log in args.logs:
        rows = parse_log(log)
        all_rows.extend(rows)
        gm = geomean_us(rows)
        if gm is not None:
            print(f"{log}: geomean_mean_us={gm:.3f}")
        for row in rows:
            print(
                "  "
                f"B{row['bs']} N{row['N']} C{row['C']} H{row['H']} "
                f"nomask={row['nomask']} {row['distribution']} "
                f"mean_us={row['mean_us']} best_us={row['best_us']} runs={row['runs']}"
            )

    if args.csv and all_rows:
        write_header = not args.csv.exists()
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(all_rows)


if __name__ == "__main__":
    main()
