#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat <<'EOF'
Usage:
  ./scripts/gcp_create_t4_spot_vm.sh PROJECT_ID ZONE VM_NAME

Example:
  ./scripts/gcp_create_t4_spot_vm.sh my-gcp-project us-east1-d cuda-t4-dev
EOF
  exit 1
fi

PROJECT_ID="$1"
ZONE="$2"
VM_NAME="$3"

gcloud config set project "${PROJECT_ID}"

gcloud compute instances create "${VM_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --machine-type "n1-standard-4" \
  --boot-disk-size "80GB" \
  --accelerator "type=nvidia-tesla-t4,count=1" \
  --image-family "common-cu129-ubuntu-2204-nvidia-580" \
  --image-project "deeplearning-platform-release" \
  --maintenance-policy "TERMINATE" \
  --provisioning-model "SPOT" \
  --instance-termination-action "STOP" \
  --no-service-account \
  --no-scopes

