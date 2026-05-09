#!/usr/bin/env bash
set -euo pipefail

INSTANCE="${INSTANCE:-cuda-a100-dev-matmul-v2}"
ZONE="${ZONE:-us-central1-a}"
REMOTE_DIR="${REMOTE_DIR:-~/matmul_v2_profile}"

gcloud compute ssh "$INSTANCE" --zone "$ZONE" --command "mkdir -p $REMOTE_DIR"
gcloud compute scp gpumode/matmul_v2/scripts/a100_profile_mm.py "$INSTANCE:$REMOTE_DIR/a100_profile_mm.py" --zone "$ZONE"
printf -v REMOTE_ARGS "%q " "$@"
gcloud compute ssh "$INSTANCE" --zone "$ZONE" --command "python3 $REMOTE_DIR/a100_profile_mm.py $REMOTE_ARGS"
