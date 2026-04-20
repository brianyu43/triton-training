#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:-nemo-488500}"
ZONE="${2:-us-east1-d}"
VM_NAME="${3:-cuda-t4-dev-lesson02}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${ROOT_DIR}/results"
LOG_PATH="${LOG_DIR}/lesson07-run-${STAMP}.log"
REMOTE_DIR="${ROOT_DIR}/results/remote"

mkdir -p "${LOG_DIR}" "${REMOTE_DIR}"

echo ">>> copying repo to VM"
"${ROOT_DIR}/scripts/gcp_copy_repo_to_vm.sh" "${PROJECT_ID}" "${ZONE}" "${VM_NAME}" 2>&1 | tee -a "${LOG_PATH}"

REMOTE_CMD='
set -euo pipefail
cd ~/cudatraining

# Clean macOS tar noise.
find . -name "._*" -delete

# Make CUDA visible to PyTorch build (nvcc + headers + libs).
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

PY="${PY:-python3}"

# Ensure pip is present (Ubuntu images sometimes ship without it).
$PY -m pip --version >/dev/null 2>&1 || sudo apt-get install -y -qq python3-pip

# Install PyTorch if missing. cu121 wheels work against CUDA 12.x runtime.
$PY -c "import torch" 2>/dev/null || {
  echo "+++ installing PyTorch (cu121 wheels)"
  $PY -m pip install --user --quiet torch --index-url https://download.pytorch.org/whl/cu121
}

# Sanity.
$PY -c "import torch; print(\"torch=\", torch.__version__, \"  cuda=\", torch.version.cuda, \"  device=\", torch.cuda.get_device_name(0))"

# Build extension in-place. TORCH_CUDA_ARCH_LIST tells PyTorch nvcc to target sm_75.
export TORCH_CUDA_ARCH_LIST="7.5"
cd extension
rm -rf build
$PY setup.py build_ext --inplace 2>&1 | tail -40
cd ..

# Correctness.
echo "+++ correctness tests"
PYTHONPATH="extension:extension/python" $PY extension/python/test_correctness.py

# Benchmark.
echo "+++ benchmark"
mkdir -p results
PYTHONPATH="extension" $PY extension/bench/bench_ops.py
PYTHONPATH="extension" $PY extension/bench/bench_ops.py --csv > results/flash_attention_torchop_t4.csv
cat results/flash_attention_torchop_t4.csv
'

echo ">>> running build + tests on VM" | tee -a "${LOG_PATH}"
gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --command "bash -lc '${REMOTE_CMD}'" 2>&1 | tee -a "${LOG_PATH}"

echo ">>> pulling results back" | tee -a "${LOG_PATH}"
gcloud compute scp \
  "${VM_NAME}:~/cudatraining/results/flash_attention_torchop_t4.csv" \
  "${REMOTE_DIR}/" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" 2>&1 | tee -a "${LOG_PATH}"

echo ">>> done" | tee -a "${LOG_PATH}"
echo "    CSV : ${REMOTE_DIR}/flash_attention_torchop_t4.csv"
echo "    log : ${LOG_PATH}"
