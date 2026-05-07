#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TRIMUL_DIR="$ROOT_DIR/gpumode/trimul_a100"
CASES="$TRIMUL_DIR/cases/v39_stage_compare_cases.txt"
LOG_DIR="$TRIMUL_DIR/logs"

run_stage() {
  local submission="$1"
  local env_value="$2"
  local before
  local after
  before="$(mktemp)"
  after="$(mktemp)"
  find "$LOG_DIR" -maxdepth 1 -type f -name '*.out' -print | sort > "$before" || true
  set +e
  REMOTE_ENV="$env_value" "$TRIMUL_DIR/scripts/gcp_eval_submission.sh" test "$submission" "$CASES" >&2
  local status="$?"
  set -e
  if [[ "$status" != "0" ]]; then
    echo "warning: $submission exited with status $status; preserving any stage log" >&2
  fi
  find "$LOG_DIR" -maxdepth 1 -type f -name '*.out' -print | sort > "$after" || true
  comm -13 "$before" "$after" | tail -1
  rm -f "$before" "$after"
}

v36_log="$(run_stage \
  "$TRIMUL_DIR/submissions/v36_cuda_ext_stage_timing_v32.py" \
  "TRIMUL_STAGE_TIMING=1 TRIMUL_STAGE_TIMING_LIMIT=16")"

v37_log="$(run_stage \
  "$TRIMUL_DIR/submissions/v37_cuda_ext_vec_gate.py" \
  "TRIMUL_STAGE_TIMING=1 TRIMUL_STAGE_TIMING_LIMIT=16")"

rank02_log="$(run_stage \
  "$TRIMUL_DIR/submissions/v39_rank02_stage_timing.py" \
  "TRIMUL_TUNE=1 TRIMUL_STAGE_PROFILE=1 TRIMUL_STAGE_EACH_CALL=1 TRIMUL_STAGE_LEARN=1")"

echo "v36_log=$v36_log"
echo "v37_log=$v37_log"
echo "rank02_log=$rank02_log"
echo
"$TRIMUL_DIR/scripts/parse_v39_stage_compare.py" "$v36_log" "$v37_log" "$rank02_log"
