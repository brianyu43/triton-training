#!/usr/bin/env bash
set -euo pipefail

# Lesson 08 · Phase 4 : Triton Flash Attention + CUDA FA from Lesson 6 baseline.

PROJECT_ID="${1:-nemo-488500}"
ZONE="${2:-us-west4-a}"
VM_NAME="${3:-cuda-l4-dev-lesson08}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${ROOT_DIR}/results"
LOG_PATH="${LOG_DIR}/lesson08-phase4-${STAMP}.log"
REMOTE_DIR="${ROOT_DIR}/results/remote"

mkdir -p "${LOG_DIR}" "${REMOTE_DIR}"

echo ">>> copying repo to VM" | tee -a "${LOG_PATH}"
"${ROOT_DIR}/scripts/gcp_copy_repo_to_vm.sh" "${PROJECT_ID}" "${ZONE}" "${VM_NAME}" 2>&1 | tee -a "${LOG_PATH}"

REMOTE_CMD='
set -euo pipefail
cd ~/cudatraining

find . -name "._*" -delete

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

PY="${PY:-python3}"

$PY -c "import torch, triton; print(\"torch=\", torch.__version__, \"  triton=\", triton.__version__, \"  device=\", torch.cuda.get_device_name(0), \"  cap=\", torch.cuda.get_device_capability(0))"

echo "+++ building CUDA flash_attention binary"
make flash_attention 2>&1 | tail -10
ls -la bin/flash_attention

echo "+++ triton flash attention bench"
mkdir -p results
$PY triton_kernels/bench/bench_flash_attention.py --csv
mv flash_attention_4way.csv results/ || true
echo "--- flash_attention_4way.csv"
cat results/flash_attention_4way.csv
'

echo ">>> running bench on VM" | tee -a "${LOG_PATH}"
gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --command "bash -lc '${REMOTE_CMD}'" 2>&1 | tee -a "${LOG_PATH}"

echo ">>> pulling CSV back" | tee -a "${LOG_PATH}"
gcloud compute scp \
  "${VM_NAME}:~/cudatraining/results/flash_attention_4way.csv" \
  "${REMOTE_DIR}/" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" 2>&1 | tee -a "${LOG_PATH}"

echo ">>> done" | tee -a "${LOG_PATH}"
echo "    log : ${LOG_PATH}"
echo "    CSV : ${REMOTE_DIR}/flash_attention_4way.csv"
