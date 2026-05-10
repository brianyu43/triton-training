#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-results/softmax.csv}"
BIN="${BIN:-./bin/softmax}"
M="${M:-4096}"

SIZES=(1024 2048 4096 8192)
VERSIONS=(v1 v2 v3)

mkdir -p "$(dirname "$OUT")"

first=1
for n in "${SIZES[@]}"; do
  for ver in "${VERSIONS[@]}"; do
    # v2 requires row to fit in 48KB shared memory; cap at N=12288
    if [ "$ver" = "v2" ] && [ "$n" -gt 12288 ]; then
      echo "skipping v2 at N=${n} (exceeds shared memory)" >&2
      continue
    fi

    echo ">>> M=${M} N=${n} version=${ver}" >&2
    output=$("$BIN" --m "$M" --n "$n" --iterations 50 --warmup 10 --version "$ver" --csv)

    if [ $first -eq 1 ]; then
      echo "$output" > "$OUT"
      first=0
    else
      echo "$output" | tail -n +2 >> "$OUT"
    fi
  done
done

echo "wrote $OUT" >&2
