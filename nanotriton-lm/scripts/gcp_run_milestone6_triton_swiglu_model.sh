#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat <<'EOF'
Usage:
  ./scripts/gcp_run_milestone6_triton_swiglu_model.sh PROJECT_ID ZONE VM_NAME [MAX_ITERS]

Example:
  ./scripts/gcp_run_milestone6_triton_swiglu_model.sh nemo-488500 us-central1-a pxr-chemprop-l4-image-run 120
EOF
  exit 1
fi

PROJECT_ID="$1"
ZONE="$2"
VM_NAME="$3"
MAX_ITERS="${4:-120}"

"$(dirname "${BASH_SOURCE[0]}")"/gcp_copy_project_to_vm.sh "${PROJECT_ID}" "${ZONE}" "${VM_NAME}"

REMOTE_CMD="$(cat <<EOF
set -euo pipefail
cd ~/nanotriton-lm
python3 -m pip install --user 'PyYAML>=6.0' 'pytest>=8.0'
python3 scripts/env_report.py
python3 data/shakespeare_char/prepare.py
python3 -m pytest -q tests/test_model_ref.py tests/test_swiglu.py
rm -rf out/loss_regression_ref out/loss_regression_triton_swiglu
python3 -m nanotriton.train --config configs/tiny_shakespeare_ref.yaml --max-iters ${MAX_ITERS} --out-dir out/loss_regression_ref
python3 -m nanotriton.train --config configs/tiny_shakespeare_triton_swiglu.yaml --max-iters ${MAX_ITERS} --out-dir out/loss_regression_triton_swiglu
python3 scripts/compare_loss_curves.py \
  --ref out/loss_regression_ref/metrics.jsonl \
  --triton out/loss_regression_triton_swiglu/metrics.jsonl \
  --max-final-val-diff 0.25
python3 -m nanotriton.generate \
  --ckpt out/loss_regression_triton_swiglu/checkpoint.pt \
  --prompt "To be" \
  --max-new-tokens 80
EOF
)"

gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --command "bash -lc $(printf '%q' "${REMOTE_CMD}")"
