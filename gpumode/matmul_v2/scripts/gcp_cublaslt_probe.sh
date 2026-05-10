#!/usr/bin/env bash
set -euo pipefail

INSTANCE="${INSTANCE:-cuda-a100-dev-matmul-v2}"
ZONE="${ZONE:-us-central1-a}"
REMOTE_DIR="${REMOTE_DIR:-~/matmul_v2_cublaslt}"

gcloud compute ssh "$INSTANCE" --zone "$ZONE" --command "mkdir -p $REMOTE_DIR"
gcloud compute scp gpumode/matmul_v2/scripts/a100_cublaslt_probe.py "$INSTANCE:$REMOTE_DIR/a100_cublaslt_probe.py" --zone "$ZONE"
printf -v REMOTE_ARGS "%q " "$@"
gcloud compute ssh "$INSTANCE" --zone "$ZONE" --command "cd $REMOTE_DIR && python3 a100_cublaslt_probe.py $REMOTE_ARGS"
