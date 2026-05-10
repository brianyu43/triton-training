#!/usr/bin/env bash
set -euo pipefail

INSTANCE="${INSTANCE:-cuda-a100-dev-matmul-v2}"
ZONE="${ZONE:-us-central1-a}"

gcloud compute instances start "$INSTANCE" --zone "$ZONE"
gcloud compute instances describe "$INSTANCE" --zone "$ZONE" \
  --format='table(name,status,machineType.basename(),guestAccelerators[].acceleratorType.basename(),guestAccelerators[].acceleratorCount)'
