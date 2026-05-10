#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat <<'EOF'
Usage:
  ./scripts/gcp_copy_project_to_vm.sh PROJECT_ID ZONE VM_NAME

Example:
  ./scripts/gcp_copy_project_to_vm.sh nemo-488500 us-east4-c cuda-l4-dev-lesson09
EOF
  exit 1
fi

PROJECT_ID="$1"
ZONE="$2"
VM_NAME="$3"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
TMP_TGZ="$(mktemp /tmp/nanotriton-lm.XXXXXX.tar.gz)"
trap 'rm -f "${TMP_TGZ}"' EXIT
rm -f "${TMP_TGZ}"

cd "${ROOT_DIR}"
export COPYFILE_DISABLE=1
tar \
  --no-xattrs \
  --exclude="./out" \
  --exclude="./.pytest_cache" \
  --exclude="./*.egg-info" \
  --exclude="./references/nanogpt" \
  --exclude="./references/triton" \
  --exclude="./references/flash-attention" \
  --exclude="./data/shakespeare_char/input.txt" \
  --exclude="./data/shakespeare_char/train.bin" \
  --exclude="./data/shakespeare_char/val.bin" \
  --exclude="./data/shakespeare_char/meta.json" \
  -czf "${TMP_TGZ}" .

gcloud compute scp "${TMP_TGZ}" "${VM_NAME}:~/nanotriton-lm.tar.gz" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}"

gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --command "rm -rf ~/nanotriton-lm && mkdir -p ~/nanotriton-lm && tar -xzf ~/nanotriton-lm.tar.gz -C ~/nanotriton-lm"
