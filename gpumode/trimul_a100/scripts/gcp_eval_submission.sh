#!/usr/bin/env bash
set -euo pipefail

INSTANCE="${INSTANCE:-cuda-a100-dev-matmul-v2}"
ZONE="${ZONE:-us-central1-a}"
MODE="${1:-benchmark}"
SUBMISSION="${2:-gpumode/trimul_a100/submissions/v01_functional_bf16.py}"
CASES="${3:-}"
REMOTE_ENV_VALUE="${REMOTE_ENV:-}"

if [[ "$MODE" != "test" && "$MODE" != "benchmark" && "$MODE" != "leaderboard" && "$MODE" != "profile" ]]; then
  echo "usage: $0 {test|benchmark|leaderboard|profile} path/to/submission.py [path/to/cases.txt]" >&2
  exit 2
fi

if [[ -z "$CASES" ]]; then
  if [[ "$MODE" == "test" ]]; then
    CASES="gpumode/trimul_a100/cases/test_cases.txt"
  else
    CASES="gpumode/trimul_a100/cases/benchmark_cases.txt"
  fi
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
SUB_BASENAME="$(basename "$SUBMISSION" .py)"
LOG_DIR="$ROOT_DIR/gpumode/trimul_a100/logs"
LOCAL_LOG="$LOG_DIR/${SUB_BASENAME}_${MODE}_${STAMP}.out"

REMOTE_ROOT="/home/xavier"
REMOTE_REPO="$REMOTE_ROOT/reference-kernels"
REMOTE_PROJ="$REMOTE_ROOT/cudatraining/gpumode/trimul_a100"
REMOTE_PROBLEM="$REMOTE_REPO/problems/bioml/trimul"
REMOTE_SUB="$REMOTE_PROJ/submissions/$(basename "$SUBMISSION")"
REMOTE_CASES="$REMOTE_PROJ/cases/$(basename "$CASES")"

mkdir -p "$LOG_DIR"

STATUS="$(gcloud compute instances describe "$INSTANCE" --zone "$ZONE" --format='value(status)')"
if [[ "$STATUS" != "RUNNING" ]]; then
  echo "VM $INSTANCE is $STATUS. Start it first with scripts/gcp_start_a100.sh" >&2
  exit 3
fi

gcloud compute ssh "$INSTANCE" --zone "$ZONE" --command \
  "mkdir -p '$REMOTE_PROJ/submissions' '$REMOTE_PROJ/cases' '$REMOTE_PROJ/logs'"

gcloud compute scp "$SUBMISSION" "$INSTANCE:$REMOTE_SUB" --zone "$ZONE"
gcloud compute scp "$CASES" "$INSTANCE:$REMOTE_CASES" --zone "$ZONE"

set +e
gcloud compute ssh "$INSTANCE" --zone "$ZONE" --command "
set -euo pipefail
export PATH=\"\$HOME/.local/bin:\$PATH\"
REMOTE_ENV='$REMOTE_ENV_VALUE'
if [ -n \"\$REMOTE_ENV\" ]; then
  export \$REMOTE_ENV
fi
cd '$REMOTE_ROOT'
if [ ! -d '$REMOTE_REPO' ]; then
  git clone --depth 1 https://github.com/gpu-mode/reference-kernels.git '$REMOTE_REPO'
else
  cd '$REMOTE_REPO' && git pull --ff-only
fi
cd '$REMOTE_PROBLEM'
cp '$REMOTE_SUB' submission.py
OUT='$REMOTE_PROJ/logs/${SUB_BASENAME}_${MODE}_${STAMP}.out'
POPCORN_FD=3 python3 eval.py '$MODE' '$REMOTE_CASES' 3> \"\$OUT\"
status=\$?
cat \"\$OUT\"
exit \$status
" 2>&1 | tee "$LOCAL_LOG"
status="${PIPESTATUS[0]}"
set -e

echo "local-log: $LOCAL_LOG"
exit "$status"
