#!/usr/bin/env python3
"""
Lesson 13 / Candidate B Stage 2 — toggle the SM-aware dispatch heuristic
in the *installed* vLLM tree.

What it flips
-------------
File: <venv>/lib/python<ver>/site-packages/vllm/v1/attention/backends/triton_attn.py
Line: the one computing `self.seq_threshold_3D` inside
      TritonAttentionMetadataBuilder.__init__

  vanilla  (upstream):
      self.seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // self.num_heads_kv

  smaware  (our candidate B patch):
      self.seq_threshold_3D = max(
          torch.cuda.get_device_properties(device).multi_processor_count // 2, 1
      ) // self.num_heads_kv  # SM-aware (lesson13 candidateB)

Why not sed
-----------
Line numbers drift across vLLM versions. We anchor on the textual pattern
`self.seq_threshold_3D = ` instead, detect the current state, and refuse to
patch if the file already has edits we don't recognize.

Usage
-----
    python scripts/toggle_vllm_patch.py --mode vanilla  [--yes]
    python scripts/toggle_vllm_patch.py --mode smaware  [--yes]
    python scripts/toggle_vllm_patch.py --status

With an active venv the script auto-locates the installed vllm; otherwise
pass `--vllm-root /path/to/site-packages/vllm`.
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


REL_TARGET = Path("v1/attention/backends/triton_attn.py")

# Anchor: the assignment line inside TritonAttentionMetadataBuilder.__init__.
# Captured with (indent, rhs) so we preserve indentation on rewrite.
# Trailing allowance is `[ \t]*`, NOT `\s*` — the latter would swallow the
# following blank line in MULTILINE mode because `\s` matches `\n`.
ANCHOR_RE = re.compile(
    r"^(?P<indent>[ \t]+)self\.seq_threshold_3D[ \t]*=[ \t]*(?P<rhs>.+?)[ \t]*$",
    re.MULTILINE,
)

VANILLA_RHS = "MIN_LAUNCH_GRID_SIZE_2D // self.num_heads_kv"
SMAWARE_RHS = (
    "max(torch.cuda.get_device_properties(device).multi_processor_count // 2, 1) "
    "// self.num_heads_kv  # SM-aware (lesson13 candidateB)"
)

# Second occurrence (inside `if self.decode_cudagraph_enabled:`) assigns from a
# `min(...)` over capture sizes — that one we MUST NOT touch. We only rewrite
# the FIRST match, which is the heuristic-boundary line.


def find_installed_vllm() -> Path:
    """Locate the vllm package directory of the currently-active Python env."""
    import_probe = subprocess.run(
        [sys.executable, "-c", "import vllm, os; print(os.path.dirname(vllm.__file__))"],
        capture_output=True,
        text=True,
    )
    if import_probe.returncode != 0:
        sys.stderr.write(
            "ERROR: could not import vllm from the active Python environment.\n"
            f"       stderr: {import_probe.stderr.strip()}\n"
            "       Activate the venv where you installed vllm, or pass --vllm-root.\n"
        )
        sys.exit(2)
    root = Path(import_probe.stdout.strip())
    if not root.is_dir():
        sys.stderr.write(f"ERROR: resolved vllm path does not exist: {root}\n")
        sys.exit(2)
    return root


def detect_state(src: str) -> tuple[str, int, str]:
    """
    Return (state, match_index, current_rhs) for the FIRST anchor match.

    state ∈ {"vanilla", "smaware", "unknown", "missing"}
    """
    match = ANCHOR_RE.search(src)
    if match is None:
        return "missing", -1, ""
    rhs = match.group("rhs").strip()
    if rhs == VANILLA_RHS:
        return "vanilla", match.start(), rhs
    if rhs == SMAWARE_RHS:
        return "smaware", match.start(), rhs
    # Anything else: either a version we don't recognize or a partial edit.
    return "unknown", match.start(), rhs


def build_new_line(indent: str, mode: str) -> str:
    rhs = VANILLA_RHS if mode == "vanilla" else SMAWARE_RHS
    return f"{indent}self.seq_threshold_3D = {rhs}"


def make_diff(old: str, new: str, path: Path) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=str(path) + " (before)",
            tofile=str(path) + " (after)",
            n=3,
        )
    )


def write_atomic(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=("vanilla", "smaware"),
        help="Target state. Omit with --status to just report current state.",
    )
    ap.add_argument(
        "--status",
        action="store_true",
        help="Report current state of the target file and exit.",
    )
    ap.add_argument(
        "--vllm-root",
        type=Path,
        default=None,
        help="Path to the installed vllm package (dir containing __init__.py). "
             "Defaults to the vllm importable from the active Python.",
    )
    ap.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation before writing.",
    )
    ap.add_argument(
        "--backup",
        action="store_true",
        help="Write a .bak copy next to the target before modifying. "
             "Off by default because the anchor is pattern-matched and "
             "git can recover the upstream line at any time.",
    )
    args = ap.parse_args()

    if not args.status and args.mode is None:
        ap.error("must pass --mode {vanilla,smaware} or --status")

    vllm_root = args.vllm_root or find_installed_vllm()
    target = vllm_root / REL_TARGET
    if not target.is_file():
        sys.stderr.write(f"ERROR: target file not found: {target}\n")
        return 2

    src = target.read_text()
    state, _, current_rhs = detect_state(src)
    print(f"[toggle] target : {target}")
    print(f"[toggle] state  : {state}")
    if state == "unknown":
        print(f"[toggle] rhs    : {current_rhs}")

    if args.status:
        return 0 if state in ("vanilla", "smaware") else 1

    if state == "missing":
        sys.stderr.write(
            "ERROR: could not find the `self.seq_threshold_3D = ...` anchor "
            "in the target file. Has the vLLM source layout changed?\n"
        )
        return 3
    if state == "unknown":
        sys.stderr.write(
            "ERROR: file already has a non-standard rhs for seq_threshold_3D — "
            "refusing to overwrite an unknown edit. Restore the upstream file "
            "(pip install --force-reinstall vllm) and re-run.\n"
        )
        return 3
    if state == args.mode:
        print(f"[toggle] already in {args.mode} state, nothing to do.")
        return 0

    # Rewrite the first match only — we anchor on the *first* hit so the
    # capture-size branch further down stays untouched.
    match = ANCHOR_RE.search(src)
    assert match is not None  # guaranteed by state != "missing"
    indent = match.group("indent")
    new_line = build_new_line(indent, args.mode)
    new_src = src[: match.start()] + new_line + src[match.end():]

    diff = make_diff(src, new_src, target)
    print("\n[toggle] proposed diff:")
    print(diff)

    if not args.yes:
        resp = input(f"[toggle] apply {state} -> {args.mode}? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            print("[toggle] aborted.")
            return 1

    if args.backup:
        shutil.copy2(target, target.with_suffix(target.suffix + ".bak"))
        print(f"[toggle] backup written: {target.with_suffix(target.suffix + '.bak')}")

    write_atomic(target, new_src)

    # Verify round-trip
    after_state, _, _ = detect_state(target.read_text())
    print(f"[toggle] new state: {after_state}")
    if after_state != args.mode:
        sys.stderr.write(
            f"ERROR: post-write state is {after_state}, expected {args.mode}.\n"
        )
        return 4

    # Pycache can cache the old bytecode. Drop the compiled .pyc for the target.
    pycache_dir = target.parent / "__pycache__"
    if pycache_dir.is_dir():
        stem = target.stem  # triton_attn
        stale = list(pycache_dir.glob(f"{stem}.cpython-*.pyc"))
        for p in stale:
            p.unlink()
            print(f"[toggle] removed stale bytecode: {p.name}")

    print(f"[toggle] ok — vllm is now in '{args.mode}' state.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
