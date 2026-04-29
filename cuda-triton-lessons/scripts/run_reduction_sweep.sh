#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-results/reduction.csv}"
BIN="${BIN:-./bin/reduction}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"

EXPONENTS=(20 22 24 26 28)
VERSIONS=(v1 v2 v3 v4 thrust)

mkdir -p "$(dirname "$OUT")"

first=1
for exp in "${EXPONENTS[@]}"; do
  n=$((1 << exp))
  for ver in "${VERSIONS[@]}"; do
    iters=50
    warmup=10
    if [ "$ver" = "v1" ]; then
      if [ "$exp" -ge 26 ]; then
        iters=5
        warmup=2
      elif [ "$exp" -ge 24 ]; then
        iters=20
        warmup=5
      fi
    fi

    echo ">>> n=2^${exp}=${n} version=${ver} iterations=${iters}" >&2
    output=$("$BIN" --n "$n" --block-size "$BLOCK_SIZE" --iterations "$iters" --warmup "$warmup" --version "$ver" --csv)

    if [ $first -eq 1 ]; then
      echo "$output" > "$OUT"
      first=0
    else
      echo "$output" | tail -n +2 >> "$OUT"
    fi
  done
done

echo "wrote $OUT" >&2
