#!/usr/bin/env bash
set -euo pipefail

# Lesson 08 · Phase 0 : Triton smoke test on L4.
# Assumes the L4 VM is already created via gcp_create_l4_spot_vm.sh
# and that PyTorch + Triton were installed on it.

PROJECT_ID="${1:-nemo-488500}"
ZONE="${2:-us-west4-a}"
VM_NAME="${3:-cuda-l4-dev-lesson08}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${ROOT_DIR}/results"
LOG_PATH="${LOG_DIR}/lesson08-phase0-${STAMP}.log"

mkdir -p "${LOG_DIR}"

echo ">>> copying repo to VM" | tee -a "${LOG_PATH}"
"${ROOT_DIR}/scripts/gcp_copy_repo_to_vm.sh" "${PROJECT_ID}" "${ZONE}" "${VM_NAME}" 2>&1 | tee -a "${LOG_PATH}"

REMOTE_CMD='
set -euo pipefail
cd ~/cudatraining

# Clean macOS tar noise.
find . -name "._*" -delete

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

PY="${PY:-python3}"

# Sanity: torch + triton should already be installed from Phase 0 setup.
$PY -c "import torch, triton; print(\"torch=\", torch.__version__, \"  cuda=\", torch.version.cuda, \"  device=\", torch.cuda.get_device_name(0), \"  cap=\", torch.cuda.get_device_capability(0)); print(\"triton=\", triton.__version__)"

echo "+++ triton smoke test (vector add)"
$PY triton_kernels/smoke_vector_add.py
'

echo ">>> running triton smoke on VM" | tee -a "${LOG_PATH}"
gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --command "bash -lc '${REMOTE_CMD}'" 2>&1 | tee -a "${LOG_PATH}"

echo ">>> done" | tee -a "${LOG_PATH}"
echo "    log : ${LOG_PATH}"
