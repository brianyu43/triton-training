#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TRIMUL_DIR="$ROOT_DIR/gpumode/trimul_a100"
SUBMISSION="${1:-$TRIMUL_DIR/submissions/v25_cuda_ext_stage_timing.py}"
CASES="${2:-$TRIMUL_DIR/cases/benchmark_cases.txt}"
LIMIT="${TRIMUL_STAGE_TIMING_LIMIT:-16}"

REMOTE_ENV="TRIMUL_STAGE_TIMING=1 TRIMUL_STAGE_TIMING_LIMIT=$LIMIT" \
  "$TRIMUL_DIR/scripts/gcp_eval_submission.sh" test "$SUBMISSION" "$CASES"
