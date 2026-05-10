#!/usr/bin/env bash
set -euo pipefail

INSTANCE="${INSTANCE:-cuda-a100-dev-matmul-v2}"
ZONE="${ZONE:-us-central1-a}"
PROBE_ARGS="$*"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/gpumode/vectorsum_v2/logs"
LOCAL_LOG="$LOG_DIR/v14_bandwidth_sass_probe_${STAMP}.out"

REMOTE_ROOT="/home/xavier"
REMOTE_PROJ="$REMOTE_ROOT/cudatraining/gpumode/vectorsum_v2"
REMOTE_SCRIPT="$REMOTE_PROJ/scripts/v14_bandwidth_sass_probe.py"

mkdir -p "$LOG_DIR"

STATUS="$(gcloud compute instances describe "$INSTANCE" --zone "$ZONE" --format='value(status)')"
if [[ "$STATUS" != "RUNNING" ]]; then
  echo "VM $INSTANCE is $STATUS. Start it first with gpumode/vectorsum_v2/scripts/gcp_start_a100.sh" >&2
  exit 3
fi

gcloud compute ssh "$INSTANCE" --zone "$ZONE" --command \
  "mkdir -p '$REMOTE_PROJ/scripts' '$REMOTE_PROJ/logs'"

gcloud compute scp "$ROOT_DIR/gpumode/vectorsum_v2/scripts/v14_bandwidth_sass_probe.py" \
  "$INSTANCE:$REMOTE_SCRIPT" \
  --zone "$ZONE"

gcloud compute ssh "$INSTANCE" --zone "$ZONE" --command "
set -euo pipefail
export PATH=\"\$HOME/.local/bin:/usr/local/cuda/bin:/usr/local/cuda-12.9/bin:\$PATH\"
cd '$REMOTE_ROOT'
python3 '$REMOTE_SCRIPT' $PROBE_ARGS
" 2>&1 | tee "$LOCAL_LOG"

echo "local-log: $LOCAL_LOG"
