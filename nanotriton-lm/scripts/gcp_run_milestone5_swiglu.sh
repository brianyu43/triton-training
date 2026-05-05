#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat <<'EOF'
Usage:
  ./scripts/gcp_run_milestone5_swiglu.sh PROJECT_ID ZONE VM_NAME

Example:
  ./scripts/gcp_run_milestone5_swiglu.sh nemo-488500 us-central1-a pxr-chemprop-l4-image-run
EOF
  exit 1
fi

PROJECT_ID="$1"
ZONE="$2"
VM_NAME="$3"

"$(dirname "${BASH_SOURCE[0]}")"/gcp_copy_project_to_vm.sh "${PROJECT_ID}" "${ZONE}" "${VM_NAME}"

REMOTE_CMD="$(cat <<'EOF'
set -euo pipefail
cd ~/nanotriton-lm
python3 -m pip install --user 'PyYAML>=6.0' 'pytest>=8.0'
python3 scripts/env_report.py
python3 -m pytest -q tests/test_swiglu.py
python3 -m benchmarks.bench_swiglu --batch 16 --seq 128 --hidden 512 --dtype float16 --warmup 20 --iters 100
EOF
)"

gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --command "bash -lc $(printf '%q' "${REMOTE_CMD}")"
