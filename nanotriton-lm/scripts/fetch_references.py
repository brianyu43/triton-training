from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Reference:
    name: str
    url: str
    commit: str
    license: str
    purpose: str


REFERENCES = {
    "nanogpt": Reference(
        name="nanogpt",
        url="https://github.com/karpathy/nanoGPT",
        commit="3adf61e154c3fe3fca428ad6bc3818b27a3b8291",
        license="MIT",
        purpose="minimal GPT training loop, config, and generation reference",
    ),
    "triton": Reference(
        name="triton",
        url="https://github.com/triton-lang/triton",
        commit="0f5f46ef80b90488f3dd9f64737ad79c3a6cafe6",
        license="MIT-style",
        purpose="Triton tutorial and kernel implementation patterns",
    ),
    "flash-attention": Reference(
        name="flash-attention",
        url="https://github.com/Dao-AILab/flash-attention",
        commit="ba59def94cd7a0c12e2a8c673b0a4655be67c5c4",
        license="BSD-3-Clause",
        purpose="attention forward/backward math and benchmarking reference",
    ),
}


def run(cmd: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True)
    return completed.stdout.strip()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def fetch_reference(reference: Reference, root: Path) -> None:
    target = root / "references" / reference.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        run(["git", "clone", "--no-tags", "--filter=blob:none", reference.url, str(target)])
    run(["git", "fetch", "--depth", "1", "origin", reference.commit], cwd=target)
    run(["git", "checkout", "--detach", reference.commit], cwd=target)
    actual = run(["git", "rev-parse", "HEAD"], cwd=target)
    if actual != reference.commit:
        raise RuntimeError(f"{reference.name} checkout mismatch: expected {reference.commit}, got {actual}")


def write_manifest(root: Path, selected: list[Reference]) -> None:
    manifest = root / "references" / "MANIFEST.md"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Reference Manifest",
        "",
        "Fetched reference repositories are local study material and are not part of NanoTriton-LM implementation code.",
        "",
        "| name | source | pinned commit | license | purpose |",
        "|---|---|---|---|---|",
    ]
    for ref in selected:
        lines.append(f"| {ref.name} | {ref.url} | `{ref.commit}` | {ref.license} | {ref.purpose} |")
    lines.append("")
    lines.append("Reference directories under `references/*/` are intentionally gitignored; rerun `python scripts/fetch_references.py --name all` to reconstruct them.")
    manifest.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch pinned external reference repositories.")
    parser.add_argument("--name", choices=[*REFERENCES.keys(), "all"], default="nanogpt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    names = list(REFERENCES) if args.name == "all" else [args.name]
    selected = [REFERENCES[name] for name in names]
    for ref in selected:
        print(f"fetching {ref.name} @ {ref.commit}")
        fetch_reference(ref, root)
    write_manifest(root, selected)
    print("wrote references/MANIFEST.md")


if __name__ == "__main__":
    main()
