#!/usr/bin/env bash
set -euo pipefail

INSTANCE="${INSTANCE:-cuda-a100-dev-matmul-v2}"
ZONE="${ZONE:-us-central1-a}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REMOTE_PROJ="/home/xavier/cudatraining/gpumode/sort_v2"
REMOTE_SCRIPT="$REMOTE_PROJ/scripts/probe_bucket_tolerance.py"

STATUS="$(gcloud compute instances describe "$INSTANCE" --zone "$ZONE" --format='value(status)')"
if [[ "$STATUS" != "RUNNING" ]]; then
  echo "VM $INSTANCE is $STATUS. Start it first with gpumode/sort_v2/scripts/gcp_start_a100.sh" >&2
  exit 3
fi

gcloud compute ssh "$INSTANCE" --zone "$ZONE" --command "mkdir -p '$REMOTE_PROJ/scripts'"
gcloud compute scp "$ROOT_DIR/gpumode/sort_v2/scripts/probe_bucket_tolerance.py" "$INSTANCE:$REMOTE_SCRIPT" --zone "$ZONE"

gcloud compute ssh "$INSTANCE" --zone "$ZONE" --command "
set -euo pipefail
export PATH=\"\$HOME/.local/bin:\$PATH\"
python3 '$REMOTE_SCRIPT' $*
"
