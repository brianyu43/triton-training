#!/usr/bin/env bash
set -euo pipefail

# Lesson 08 · Phase 1 : Triton reduction + autotune sweep + CUDA-on-L4 baseline.

PROJECT_ID="${1:-nemo-488500}"
ZONE="${2:-us-west4-a}"
VM_NAME="${3:-cuda-l4-dev-lesson08}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${ROOT_DIR}/results"
LOG_PATH="${LOG_DIR}/lesson08-phase1-${STAMP}.log"
REMOTE_DIR="${ROOT_DIR}/results/remote"

mkdir -p "${LOG_DIR}" "${REMOTE_DIR}"

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

# Sanity.
$PY -c "import torch, triton; print(\"torch=\", torch.__version__, \"  cuda=\", torch.version.cuda, \"  device=\", torch.cuda.get_device_name(0), \"  cap=\", torch.cuda.get_device_capability(0)); print(\"triton=\", triton.__version__)"

# Rebuild CUDA reduction for sm_89 (the Makefile already includes sm_89 gencode).
echo "+++ building CUDA reduction binary"
make clean >/dev/null 2>&1 || true
make reduction 2>&1 | tail -20
ls -la bin/reduction

# Run the 3-way bench + write CSVs.
echo "+++ triton reduction bench (sweep + 3-way)"
mkdir -p results
$PY triton_kernels/bench/bench_reduction.py --csv
mv reduction_triton_sweep.csv results/ || true
mv reduction_3way.csv         results/ || true
echo "--- reduction_triton_sweep.csv"
cat results/reduction_triton_sweep.csv
echo "--- reduction_3way.csv"
cat results/reduction_3way.csv
'

echo ">>> running build + bench on VM" | tee -a "${LOG_PATH}"
gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --command "bash -lc '${REMOTE_CMD}'" 2>&1 | tee -a "${LOG_PATH}"

echo ">>> pulling CSVs back" | tee -a "${LOG_PATH}"
gcloud compute scp \
  "${VM_NAME}:~/cudatraining/results/reduction_triton_sweep.csv" \
  "${VM_NAME}:~/cudatraining/results/reduction_3way.csv" \
  "${REMOTE_DIR}/" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" 2>&1 | tee -a "${LOG_PATH}"

echo ">>> done" | tee -a "${LOG_PATH}"
echo "    log : ${LOG_PATH}"
echo "    CSV : ${REMOTE_DIR}/reduction_triton_sweep.csv"
echo "    CSV : ${REMOTE_DIR}/reduction_3way.csv"
