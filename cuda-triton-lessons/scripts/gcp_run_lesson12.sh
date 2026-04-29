#!/usr/bin/env bash
set -euo pipefail

# Lesson 12 · split-k paged attention on L4.
#
# 1) sync repo to VM
# 2) correctness: bench_paged_attention.py  (must pass single-pass + split-k)
# 3) speed:       bench_paged_attention_speed.py --compare-paths
#                 (compares SP vs SK vs auto at block_size=16)
# 4) pull logs back under results/lesson12/

PROJECT_ID="${1:-nemo-488500}"
ZONE="${2:-us-west1-b}"
VM_NAME="${3:-cuda-l4-dev-lesson10}"
MODE="${4:-all}"   # "correctness" | "speed" | "all"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${ROOT_DIR}/results"
LOG_PATH="${LOG_DIR}/lesson12-${MODE}-${STAMP}.log"

mkdir -p "${LOG_DIR}"

echo ">>> copying repo to VM" | tee -a "${LOG_PATH}"
"${ROOT_DIR}/scripts/gcp_copy_repo_to_vm.sh" "${PROJECT_ID}" "${ZONE}" "${VM_NAME}" 2>&1 | tee -a "${LOG_PATH}"

REMOTE_PREP='
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
cd ~/cudatraining
find . -name "._*" -delete 2>/dev/null || true

echo "=== GPU + versions ==="
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
python3 -c "import torch, triton; print(\"torch\", torch.__version__); print(\"triton\", triton.__version__)"
'

CORRECTNESS_CMD="${REMOTE_PREP}"'
echo
echo "=== Lesson 12 correctness (single-pass + split-k) ==="
python3 triton_kernels/bench/bench_paged_attention.py
'

SPEED_CMD="${REMOTE_PREP}"'
echo
echo "=== Lesson 12 speed bench with --compare-paths ==="
python3 triton_kernels/bench/bench_paged_attention_speed.py --dtype fp16 --warmup 50 --iters 200 --compare-paths
'

run_remote() {
  local cmd="$1"
  gcloud compute ssh "${VM_NAME}" \
    --project "${PROJECT_ID}" \
    --zone "${ZONE}" \
    --command "bash -lc '${cmd}'" 2>&1 | tee -a "${LOG_PATH}"
}

if [[ "${MODE}" == "correctness" || "${MODE}" == "all" ]]; then
  echo ">>> running correctness bench" | tee -a "${LOG_PATH}"
  run_remote "${CORRECTNESS_CMD}"
fi

if [[ "${MODE}" == "speed" || "${MODE}" == "all" ]]; then
  echo ">>> running speed bench (path compare)" | tee -a "${LOG_PATH}"
  run_remote "${SPEED_CMD}"
fi

echo ">>> done" | tee -a "${LOG_PATH}"
echo "    log: ${LOG_PATH}"
