#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


FIELD_RE = re.compile(r"([A-Za-z_]+)=([^ ]+)")


def parse_float(fields: dict[str, str], key: str, default: float = 0.0) -> float:
    value = fields.get(key)
    if value is None:
        return default
    return float(value.rstrip(","))


def parse_int(fields: dict[str, str], key: str, default: int = 0) -> int:
    value = fields.get(key)
    if value is None:
        return default
    return int(value.rstrip(","))


def variant_from_path(path: Path) -> str:
    name = path.name
    if "v49" in name:
        return "v49"
    if "v48" in name:
        return "v48"
    if "v46" in name:
        return "v46_hybrid"
    if "v45" in name:
        return "v45"
    if "v44" in name:
        return "v44"
    if "v43" in name:
        return "v43"
    if "v42" in name:
        return "v42"
    if "rank01" in name or "v41_rank01" in name:
        return "rank01_v41"
    if "v40" in name:
        return "v40"
    if "rank02" in name or "v39_rank02" in name:
        return "rank02_v39"
    if "v37" in name:
        return "v37"
    if "v36" in name:
        return "v36_v32"
    if "v32" in name:
        return "v32"
    return path.stem


def rows_from_log(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    variant = variant_from_path(path)
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("trimul_stage_v48 "):
            fields = dict(FIELD_RE.findall(line))
            prep = parse_float(fields, "prep_ms")
            proj_gate = parse_float(fields, "proj_gate_ms")
            tail = parse_float(fields, "tail_ms")
            variant_name = variant
            path_name = fields.get("path")
            if path_name:
                variant_name = f"{variant}:{path_name}"
            rows.append(
                {
                    "variant": variant_name,
                    "shape": shape_label(fields),
                    "B": parse_int(fields, "B"),
                    "N": parse_int(fields, "N"),
                    "C": parse_int(fields, "C"),
                    "H": parse_int(fields, "H"),
                    "ln": parse_float(fields, "ln_ms"),
                    "proj_gate": prep + proj_gate,
                    "central": 0.0,
                    "hidden": 0.0,
                    "out": tail,
                    "hidden_out": tail,
                    "total": parse_float(fields, "total_ms"),
                    "source": str(path),
                }
            )
        elif line.startswith("trimul_hybrid_stage_v46 ") or line.startswith("trimul_rank01_stage_v41 "):
            fields = dict(FIELD_RE.findall(line))
            prep = parse_float(fields, "prep_ms")
            proj_gate = parse_float(fields, "proj_gate_ms")
            hidden_out = parse_float(fields, "hidden_out_ms")
            variant_name = variant
            path_name = fields.get("path")
            if path_name:
                variant_name = f"{variant}:{path_name}"
            rows.append(
                {
                    "variant": variant_name,
                    "shape": shape_label(fields),
                    "B": parse_int(fields, "B"),
                    "N": parse_int(fields, "N"),
                    "C": parse_int(fields, "C"),
                    "H": parse_int(fields, "H"),
                    "ln": parse_float(fields, "ln_ms"),
                    "proj_gate": prep + proj_gate,
                    "central": parse_float(fields, "central_ms"),
                    "hidden": 0.0,
                    "out": hidden_out,
                    "hidden_out": hidden_out,
                    "total": parse_float(fields, "total_ms"),
                    "source": str(path),
                }
            )
        elif line.startswith("trimul_rank02_stage_v39 "):
            fields = dict(FIELD_RE.findall(line))
            hidden = parse_float(fields, "hidden_ms")
            out = parse_float(fields, "out_ms")
            rows.append(
                {
                    "variant": variant,
                    "shape": shape_label(fields),
                    "B": parse_int(fields, "B"),
                    "N": parse_int(fields, "N"),
                    "C": parse_int(fields, "C"),
                    "H": parse_int(fields, "H"),
                    "ln": parse_float(fields, "ln_ms"),
                    "proj_gate": parse_float(fields, "proj_gate_ms"),
                    "central": parse_float(fields, "central_ms"),
                    "hidden": hidden,
                    "out": out,
                    "hidden_out": hidden + out,
                    "total": parse_float(fields, "total_ms"),
                    "source": str(path),
                }
            )
        elif line.startswith("trimul_stage_v"):
            fields = dict(FIELD_RE.findall(line))
            pack = parse_float(fields, "pack_ms")
            proj = parse_float(fields, "proj_ms")
            gate = parse_float(fields, "gate_ms")
            hidden = parse_float(fields, "hidden_ms")
            out = parse_float(fields, "out_ms")
            rows.append(
                {
                    "variant": variant,
                    "shape": shape_label(fields),
                    "B": parse_int(fields, "B"),
                    "N": parse_int(fields, "N"),
                    "C": parse_int(fields, "C"),
                    "H": parse_int(fields, "H"),
                    "ln": parse_float(fields, "ln_ms"),
                    "proj_gate": pack + proj + gate,
                    "central": parse_float(fields, "central_ms"),
                    "hidden": hidden,
                    "out": out,
                    "hidden_out": hidden + out,
                    "total": parse_float(fields, "total_ms"),
                    "source": str(path),
                }
            )
    return rows


def shape_label(fields: dict[str, str]) -> str:
    return f"B{fields.get('B')} N{fields.get('N')} C{fields.get('C')}"


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--all", action="store_true", help="Show every parsed row instead of the last row per shape/variant")
    args = parser.parse_args()

    parsed: list[dict[str, object]] = []
    for log in args.logs:
        parsed.extend(rows_from_log(log))

    if not args.all:
        last: dict[tuple[str, str], dict[str, object]] = {}
        for row in parsed:
            last[(str(row["variant"]), str(row["shape"]))] = row
        parsed = list(last.values())

    parsed.sort(key=lambda r: (int(r["N"]), int(r["C"]), str(r["variant"])))

    print("| Variant | Shape | LN | Proj+Gate | Central | Hidden | Out | Hidden+Out | Total |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in parsed:
        print(
            "| "
            f"{row['variant']} | {row['shape']} | "
            f"{fmt(float(row['ln']))} | {fmt(float(row['proj_gate']))} | "
            f"{fmt(float(row['central']))} | {fmt(float(row['hidden']))} | "
            f"{fmt(float(row['out']))} | {fmt(float(row['hidden_out']))} | "
            f"{fmt(float(row['total']))} |"
        )

    by_shape: dict[str, dict[str, dict[str, object]]] = {}
    for row in parsed:
        by_shape.setdefault(str(row["shape"]), {})[str(row["variant"])] = row

    print()
    print("| Shape | Base | Rank02 | Total Gap | Biggest Stage Gap |")
    print("| --- | --- | --- | ---: | --- |")
    for shape, variants in sorted(by_shape.items()):
        rank = variants.get("rank02_v39")
        base = variants.get("v37") or variants.get("v36_v32") or variants.get("v32")
        if not rank or not base:
            continue
        gaps = {
            "ln": float(base["ln"]) - float(rank["ln"]),
            "proj_gate": float(base["proj_gate"]) - float(rank["proj_gate"]),
            "central": float(base["central"]) - float(rank["central"]),
            "hidden": float(base["hidden"]) - float(rank["hidden"]),
            "out": float(base["out"]) - float(rank["out"]),
        }
        stage, gap = max(gaps.items(), key=lambda item: item[1])
        total_gap = float(base["total"]) - float(rank["total"])
        print(f"| {shape} | {base['variant']} | rank02_v39 | {total_gap:.4f} | {stage} {gap:.4f} |")


if __name__ == "__main__":
    main()
