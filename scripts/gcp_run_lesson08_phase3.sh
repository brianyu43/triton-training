#!/usr/bin/env bash
set -euo pipefail

# Lesson 08 · Phase 3 : Triton matmul + CUDA-on-L4 v3/v4 baselines.

PROJECT_ID="${1:-nemo-488500}"
ZONE="${2:-us-west4-a}"
VM_NAME="${3:-cuda-l4-dev-lesson08}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${ROOT_DIR}/results"
LOG_PATH="${LOG_DIR}/lesson08-phase3-${STAMP}.log"
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

echo "+++ building CUDA matmul binary"
make matmul 2>&1 | tail -10
ls -la bin/matmul

echo "+++ triton matmul bench (fp32 + fp16)"
mkdir -p results
$PY triton_kernels/bench/bench_matmul.py --csv
mv matmul_3way_fp32.csv results/ || true
mv matmul_3way_fp16.csv results/ || true
echo "--- matmul_3way_fp32.csv"
cat results/matmul_3way_fp32.csv
echo "--- matmul_3way_fp16.csv"
cat results/matmul_3way_fp16.csv
'

echo ">>> running bench on VM" | tee -a "${LOG_PATH}"
gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --command "bash -lc '${REMOTE_CMD}'" 2>&1 | tee -a "${LOG_PATH}"

echo ">>> pulling CSVs back" | tee -a "${LOG_PATH}"
gcloud compute scp \
  "${VM_NAME}:~/cudatraining/results/matmul_3way_fp32.csv" \
  "${VM_NAME}:~/cudatraining/results/matmul_3way_fp16.csv" \
  "${REMOTE_DIR}/" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" 2>&1 | tee -a "${LOG_PATH}"

echo ">>> done" | tee -a "${LOG_PATH}"
echo "    log : ${LOG_PATH}"
echo "    CSV : ${REMOTE_DIR}/matmul_3way_fp32.csv"
echo "    CSV : ${REMOTE_DIR}/matmul_3way_fp16.csv"
