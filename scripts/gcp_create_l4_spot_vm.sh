#!/usr/bin/env bash
set -euo pipefail

# Create an L4 (Ada Lovelace, sm_89) spot VM on GCP.
# L4 is attached via the G2 machine family, not via a separate --accelerator flag.
# g2-standard-4 = 4 vCPU, 16GB RAM, 1 L4 GPU (24GB VRAM).

if [[ $# -lt 3 ]]; then
  cat <<'EOF'
Usage:
  ./scripts/gcp_create_l4_spot_vm.sh PROJECT_ID ZONE VM_NAME

Example:
  ./scripts/gcp_create_l4_spot_vm.sh nemo-488500 us-east1-d cuda-l4-dev-lesson08
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
  --machine-type "g2-standard-4" \
  --boot-disk-size "120GB" \
  --image-family "common-cu129-ubuntu-2204-nvidia-580" \
  --image-project "deeplearning-platform-release" \
  --maintenance-policy "TERMINATE" \
  --provisioning-model "SPOT" \
  --instance-termination-action "STOP" \
  --no-service-account \
  --no-scopes
