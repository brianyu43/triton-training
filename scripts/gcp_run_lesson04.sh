#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:-nemo-488500}"
ZONE="${2:-us-east1-d}"
VM_NAME="${3:-cuda-t4-dev-lesson02}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${ROOT_DIR}/results"
LOG_PATH="${LOG_DIR}/lesson04-run-${STAMP}.log"
REMOTE_DIR="${ROOT_DIR}/results/remote"

mkdir -p "${LOG_DIR}" "${REMOTE_DIR}"

echo ">>> copying repo to VM"
"${ROOT_DIR}/scripts/gcp_copy_repo_to_vm.sh" "${PROJECT_ID}" "${ZONE}" "${VM_NAME}" 2>&1 | tee -a "${LOG_PATH}"

echo ">>> building and running matmul sweep on VM" | tee -a "${LOG_PATH}"
gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --command "bash -lc 'set -euo pipefail; cd ~/cudatraining; find . -name \"._*\" -delete; mkdir -p results bin; /usr/local/cuda/bin/nvcc -O3 -std=c++17 -lineinfo -gencode arch=compute_75,code=sm_75 src/matmul.cu -o bin/matmul; chmod +x ./scripts/run_matmul_sweep.sh; ./scripts/run_matmul_sweep.sh results/matmul_t4.csv'" 2>&1 | tee -a "${LOG_PATH}"

echo ">>> pulling results back" | tee -a "${LOG_PATH}"
gcloud compute scp \
  "${VM_NAME}:~/cudatraining/results/matmul_t4.csv" \
  "${REMOTE_DIR}/" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" 2>&1 | tee -a "${LOG_PATH}"

echo ">>> done" | tee -a "${LOG_PATH}"
echo "    CSV : ${REMOTE_DIR}/matmul_t4.csv"
echo "    log : ${LOG_PATH}"
