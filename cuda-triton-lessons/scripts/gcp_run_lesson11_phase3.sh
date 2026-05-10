#!/usr/bin/env bash
set -euo pipefail

# Lesson 11 · Phase 3 : paged attention speed bench + ncu drill on L4.
#
# 1) sync repo
# 2) run bench_paged_attention_speed.py → markdown table
# 3) (optional) ncu drill at one shape for paged + sdpa
# 4) pull logs back
#
# NOTE: ncu needs sudo on this VM family (same as Lesson 10 Phase 3).

PROJECT_ID="${1:-nemo-488500}"
ZONE="${2:-us-west1-b}"
VM_NAME="${3:-cuda-l4-dev-lesson10}"
DO_NCU="${4:-speed}"   # "speed" = bench only; "ncu" = also run ncu drill

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${ROOT_DIR}/results"
LOG_PATH="${LOG_DIR}/lesson11-phase3-${STAMP}.log"
PULL_DIR="${ROOT_DIR}/results/lesson11_phase3"

mkdir -p "${LOG_DIR}" "${PULL_DIR}"

echo ">>> copying repo to VM" | tee -a "${LOG_PATH}"
"${ROOT_DIR}/scripts/gcp_copy_repo_to_vm.sh" "${PROJECT_ID}" "${ZONE}" "${VM_NAME}" 2>&1 | tee -a "${LOG_PATH}"

REMOTE_CMD='
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
cd ~/cudatraining
find . -name "._*" -delete 2>/dev/null || true

echo "=== GPU + versions ==="
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
python3 -c "import torch, triton; print(\"torch\", torch.__version__); print(\"triton\", triton.__version__)"

echo
echo "=== Phase 3 speed bench (SDPA vs paged, block_size sweep) ==="
python3 triton_kernels/bench/bench_paged_attention_speed.py --dtype fp16 --warmup 50 --iters 200
'

NCU_CMD='
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
cd ~/cudatraining
mkdir -p ~/lesson11/phase3
cd ~/lesson11/phase3

NCU_SHAPE_ARGS="--B 8 --H 32 --H-kv 8 --d 128 --ctx 2048 --block-size 16 --warmup 20 --iterations 5"

# list kernels first so we know what to target
echo "=== list kernels (paged) ==="
python3 ~/cudatraining/triton_kernels/bench/lesson11_ncu_profile.py --mode paged ${NCU_SHAPE_ARGS} --list-kernels || true
echo "=== list kernels (sdpa) ==="
python3 ~/cudatraining/triton_kernels/bench/lesson11_ncu_profile.py --mode sdpa ${NCU_SHAPE_ARGS} --list-kernels || true

echo
echo "=== ncu drill (paged) ==="
sudo -E ncu \
  --launch-skip 20 --launch-count 1 \
  --section SpeedOfLight \
  --section WarpStateStats \
  --section LaunchStats \
  --section MemoryWorkloadAnalysis \
  --export ncu_paged --force-overwrite \
  python3 ~/cudatraining/triton_kernels/bench/lesson11_ncu_profile.py \
    --mode paged ${NCU_SHAPE_ARGS} 2>&1 | tail -120 || true

echo
echo "=== ncu drill (sdpa) ==="
sudo -E ncu \
  --launch-skip 20 --launch-count 1 \
  --section SpeedOfLight \
  --section WarpStateStats \
  --section LaunchStats \
  --section MemoryWorkloadAnalysis \
  --export ncu_sdpa --force-overwrite \
  python3 ~/cudatraining/triton_kernels/bench/lesson11_ncu_profile.py \
    --mode sdpa ${NCU_SHAPE_ARGS} 2>&1 | tail -120 || true

ls -la ~/lesson11/phase3/
'

echo ">>> running speed bench" | tee -a "${LOG_PATH}"
gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --command "bash -lc '${REMOTE_CMD}'" 2>&1 | tee -a "${LOG_PATH}"

if [[ "${DO_NCU}" == "ncu" ]]; then
  echo ">>> running ncu drill" | tee -a "${LOG_PATH}"
  gcloud compute ssh "${VM_NAME}" \
    --project "${PROJECT_ID}" \
    --zone "${ZONE}" \
    --command "bash -lc '${NCU_CMD}'" 2>&1 | tee -a "${LOG_PATH}"

  echo ">>> pulling ncu reports" | tee -a "${LOG_PATH}"
  for name in paged sdpa; do
    gcloud compute scp \
      --project "${PROJECT_ID}" --zone "${ZONE}" \
      "${VM_NAME}:~/lesson11/phase3/ncu_${name}.ncu-rep" \
      "${PULL_DIR}/ncu_${name}.ncu-rep" 2>&1 | tee -a "${LOG_PATH}" || true
  done
fi

echo ">>> done" | tee -a "${LOG_PATH}"
echo "    log: ${LOG_PATH}"
[[ "${DO_NCU}" == "ncu" ]] && echo "    ncu: ${PULL_DIR}/ncu_{paged,sdpa}.ncu-rep"
