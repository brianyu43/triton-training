#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-results/flash_attention.csv}"
BIN="${BIN:-./bin/flash_attention}"

# Sequence lengths.  Naive caps at N <= 12288 (48KB smem row); Flash has no cap.
SIZES=(512 1024 2048 4096)

mkdir -p "$(dirname "$OUT")"

first=1
for n in "${SIZES[@]}"; do
  for ver in naive flash; do
    iters=50
    # CPU reference is expensive — only use --no-check past N=2048.
    extra=""
    if [ "$n" -gt 2048 ]; then
      extra="--no-check"
    fi

    echo ">>> N=${n} version=${ver}" >&2
    output=$("$BIN" --n "$n" --iterations "$iters" --warmup 10 \
      --version "$ver" $extra --csv)

    if [ $first -eq 1 ]; then
      echo "$output" > "$OUT"
      first=0
    else
      echo "$output" | tail -n +2 >> "$OUT"
    fi
  done
done

echo "wrote $OUT" >&2
