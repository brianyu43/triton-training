#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat <<'EOF'
Usage:
  ./scripts/gcp_copy_repo_to_vm.sh PROJECT_ID ZONE VM_NAME

Example:
  ./scripts/gcp_copy_repo_to_vm.sh my-gcp-project us-east1-d cuda-t4-dev
EOF
  exit 1
fi

PROJECT_ID="$1"
ZONE="$2"
VM_NAME="$3"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
TMP_TGZ="$(mktemp /tmp/cudatraining.XXXXXX.tar.gz)"
trap 'rm -f "${TMP_TGZ}"' EXIT
rm -f "${TMP_TGZ}"

cd "${ROOT_DIR}"
export COPYFILE_DISABLE=1
tar \
  --exclude="./bin" \
  --exclude="./build" \
  --exclude="./results" \
  --exclude="./.git" \
  -czf "${TMP_TGZ}" .

gcloud compute scp "${TMP_TGZ}" "${VM_NAME}:~/cudatraining.tar.gz" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}"

gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --command "rm -rf ~/cudatraining && mkdir -p ~/cudatraining && tar -xzf ~/cudatraining.tar.gz -C ~/cudatraining"
