#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-benchmark}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TRIMUL_DIR="$ROOT_DIR/gpumode/trimul_a100"

SUBMISSIONS=(
  "$TRIMUL_DIR/submissions/v00_sample.py"
  "$TRIMUL_DIR/submissions/v01_functional_bf16.py"
  "$TRIMUL_DIR/submissions/v02_concat_bmm_fp16.py"
  "$TRIMUL_DIR/submissions/v10_hf_triton_a100.py"
)

for sub in "${SUBMISSIONS[@]}"; do
  echo "== $MODE $(basename "$sub") =="
  "$TRIMUL_DIR/scripts/gcp_eval_submission.sh" "$MODE" "$sub"
done

python3 "$TRIMUL_DIR/scripts/parse_popcorn_log.py" \
  "$TRIMUL_DIR"/logs/*_"$MODE"_*.out \
  --csv "$TRIMUL_DIR/logs/benchmark_results.csv"
