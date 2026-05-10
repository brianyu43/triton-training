#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-results/matmul.csv}"
BIN="${BIN:-./bin/matmul}"

SIZES=(256 512 1024 2048)
VERSIONS=(v1 v2 v3 v4)

mkdir -p "$(dirname "$OUT")"

first=1
for size in "${SIZES[@]}"; do
  for ver in "${VERSIONS[@]}"; do
    iters=20
    warmup=5
    # v1 at 2048 gets slow (~300ms/iter); keep iterations modest
    if [ "$ver" = "v1" ] && [ "$size" -ge 2048 ]; then
      iters=10
      warmup=3
    fi

    echo ">>> m=n=k=${size} version=${ver} iterations=${iters}" >&2
    output=$("$BIN" --m "$size" --n "$size" --k "$size" --iterations "$iters" --warmup "$warmup" --version "$ver" --csv)

    if [ $first -eq 1 ]; then
      echo "$output" > "$OUT"
      first=0
    else
      echo "$output" | tail -n +2 >> "$OUT"
    fi
  done
done

echo "wrote $OUT" >&2
