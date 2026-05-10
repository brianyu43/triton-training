#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat <<'EOF'
Usage:
  ./scripts/gcp_run_milestone1_smoke.sh PROJECT_ID ZONE VM_NAME [MAX_ITERS]

Example:
  ./scripts/gcp_run_milestone1_smoke.sh nemo-488500 us-east4-c cuda-l4-dev-lesson09 20
EOF
  exit 1
fi

PROJECT_ID="$1"
ZONE="$2"
VM_NAME="$3"
MAX_ITERS="${4:-20}"

"$(dirname "${BASH_SOURCE[0]}")"/gcp_copy_project_to_vm.sh "${PROJECT_ID}" "${ZONE}" "${VM_NAME}"

REMOTE_CMD="$(cat <<EOF
set -euo pipefail
cd ~/nanotriton-lm
python3 -m pip install --user 'PyYAML>=6.0' 'pytest>=8.0'
python3 scripts/env_report.py
python3 scripts/fetch_references.py --name all
python3 data/shakespeare_char/prepare.py
python3 -m pytest -q
python3 -m nanotriton.train --config configs/tiny_shakespeare_ref.yaml --max-iters ${MAX_ITERS}
python3 -m nanotriton.generate --ckpt out/tiny_shakespeare_ref/checkpoint.pt --prompt "To be" --max-new-tokens 80
EOF
)"

gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --command "bash -lc $(printf '%q' "${REMOTE_CMD}")"
