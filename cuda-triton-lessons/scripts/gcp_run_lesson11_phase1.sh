#!/usr/bin/env bash
set -euo pipefail

# Lesson 11 · Phase 1 : Triton paged attention (decode, MHA) correctness on L4.
#
# 1) sync repo to VM
# 2) run triton_kernels/bench/bench_paged_attention.py (fp16 + fp32, 10 shapes)
# 3) pull the log back for the handoff

PROJECT_ID="${1:-nemo-488500}"
ZONE="${2:-us-west1-b}"
VM_NAME="${3:-cuda-l4-dev-lesson10}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${ROOT_DIR}/results"
LOG_PATH="${LOG_DIR}/lesson11-phase1-${STAMP}.log"
PULL_DIR="${ROOT_DIR}/results/lesson11_phase1"

mkdir -p "${LOG_DIR}" "${PULL_DIR}"

echo ">>> copying repo to VM" | tee -a "${LOG_PATH}"
"${ROOT_DIR}/scripts/gcp_copy_repo_to_vm.sh" "${PROJECT_ID}" "${ZONE}" "${VM_NAME}" 2>&1 | tee -a "${LOG_PATH}"

REMOTE_CMD='
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
cd ~/cudatraining
find . -name "._*" -delete 2>/dev/null || true

echo "=== GPU + torch/triton versions ==="
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
python3 -c "import torch, triton; print(\"torch\", torch.__version__); print(\"triton\", triton.__version__); print(\"cuda\", torch.version.cuda)"

echo
echo "=== Phase 1 correctness (Triton paged attention vs references) ==="
python3 triton_kernels/bench/bench_paged_attention.py
'

echo ">>> running on VM" | tee -a "${LOG_PATH}"
gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --command "bash -lc '${REMOTE_CMD}'" 2>&1 | tee -a "${LOG_PATH}"

echo ">>> done" | tee -a "${LOG_PATH}"
echo "    log: ${LOG_PATH}"
