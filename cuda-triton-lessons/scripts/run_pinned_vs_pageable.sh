#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-results/pinned_vs_pageable.csv}"
BIN="${BIN:-./bin/vector_add}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
ITERATIONS="${ITERATIONS:-100}"

EXPONENTS=(14 16 18 20 22 24 26 28)
MODES=(pinned pageable)

mkdir -p "$(dirname "$OUT")"

first=1
for exp in "${EXPONENTS[@]}"; do
  n=$((1 << exp))
  for mode in "${MODES[@]}"; do
    flag=""
    if [ "$mode" = "pageable" ]; then
      flag="--pageable"
    fi

    echo ">>> running n=2^${exp}=${n} mode=${mode}" >&2
    output=$("$BIN" --n "$n" --block-size "$BLOCK_SIZE" --iterations "$ITERATIONS" $flag --csv)

    if [ $first -eq 1 ]; then
      echo "$output" > "$OUT"
      first=0
    else
      echo "$output" | tail -n +2 >> "$OUT"
    fi
  done
done

echo "wrote $OUT" >&2
