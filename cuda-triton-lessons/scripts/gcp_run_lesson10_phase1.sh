#!/usr/bin/env bash
set -euo pipefail

# Lesson 10 · Phase 1 : nsys timeline diff for lesson 04 (pinned vs pageable).
#
# Runs ./bin/vector_add twice under nsys — once with --pageable, once with
# --pinned — captures .nsys-rep files plus a text summary, and pulls them
# back to the local ./results/lesson10_phase1/ directory for GUI viewing.

PROJECT_ID="${1:-nemo-488500}"
ZONE="${2:-us-west1-b}"
VM_NAME="${3:-cuda-l4-dev-lesson10}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${ROOT_DIR}/results"
LOG_PATH="${LOG_DIR}/lesson10-phase1-${STAMP}.log"
PULL_DIR="${ROOT_DIR}/results/lesson10_phase1"

mkdir -p "${LOG_DIR}" "${PULL_DIR}"

echo ">>> copying repo to VM" | tee -a "${LOG_PATH}"
"${ROOT_DIR}/scripts/gcp_copy_repo_to_vm.sh" "${PROJECT_ID}" "${ZONE}" "${VM_NAME}" 2>&1 | tee -a "${LOG_PATH}"

REMOTE_CMD='
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
cd ~/cudatraining

find . -name "._*" -delete 2>/dev/null || true

# Build (idempotent; re-link only if source changed).
make vector_add >/dev/null

mkdir -p ~/lesson10/phase1
cd ~/lesson10/phase1

N=16777216   # 64 MB fp32 -- comfortably above L2, under GPU mem.
ITERS=5      # small enough to read the timeline, big enough to stabilize.

echo "=== sm_89 + Nsight versions ==="
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
nsys --version
ncu --version | head -1

for MODE in pageable pinned; do
  echo "=== nsys profile · $MODE ==="
  # --trace=cuda,nvtx : CUDA API + kernel timeline. Add osrt if OS calls needed.
  # --force-overwrite=true : replace prior runs.
  # --output : stem; nsys adds .nsys-rep
  nsys profile \
    --trace=cuda,nvtx \
    --force-overwrite=true \
    --output "vector_add_${MODE}" \
    ~/cudatraining/bin/vector_add \
       --n ${N} --iterations ${ITERS} --${MODE}

  # CLI summary that is readable over SSH (no GUI needed).
  # All reports rolled into one call so the sqlite export runs once, and
  # tolerated on failure so the script keeps going on either mode.
  nsys stats --force-export=true --format csv \
    --report cuda_api_sum \
    --report cuda_gpu_kern_sum \
    --report cuda_gpu_mem_time_sum \
    --report cuda_gpu_mem_size_sum \
    "vector_add_${MODE}.nsys-rep" || true
  echo
done

ls -la ~/lesson10/phase1/
'

echo ">>> running on VM" | tee -a "${LOG_PATH}"
gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --command "bash -lc '${REMOTE_CMD}'" 2>&1 | tee -a "${LOG_PATH}"

echo ">>> pulling .nsys-rep back to ${PULL_DIR}" | tee -a "${LOG_PATH}"
for MODE in pageable pinned; do
  gcloud compute scp \
    --project "${PROJECT_ID}" --zone "${ZONE}" \
    "${VM_NAME}:~/lesson10/phase1/vector_add_${MODE}.nsys-rep" \
    "${PULL_DIR}/vector_add_${MODE}.nsys-rep" 2>&1 | tee -a "${LOG_PATH}"
done

echo ">>> done" | tee -a "${LOG_PATH}"
echo "    log         : ${LOG_PATH}"
echo "    nsys reports: ${PULL_DIR}/vector_add_{pageable,pinned}.nsys-rep"
echo
echo "Open in Nsight Systems GUI:"
echo "    nsys-ui ${PULL_DIR}/vector_add_pinned.nsys-rep &"
echo "Or on Mac:  open -a 'Nsight Systems' ${PULL_DIR}/vector_add_pinned.nsys-rep"
