#!/usr/bin/env bash
set -euo pipefail

# Lesson 10 · Phase 2 : ncu stall-reason + SOL diff for reduction v1 vs v4.
#
# v1 = atomicAdd per thread  (every thread contends on a single global counter)
# v4 = warp shuffle + final atomicAdd  (one atomic per block)
#
# We expect:
#   - v1 : Stall = "Wait" (atomic serialization), very low SM throughput
#   - v4 : Stall = mostly "Selected" / "Short Scoreboard", high memory throughput

PROJECT_ID="${1:-nemo-488500}"
ZONE="${2:-us-west1-b}"
VM_NAME="${3:-cuda-l4-dev-lesson10}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${ROOT_DIR}/results"
LOG_PATH="${LOG_DIR}/lesson10-phase2-${STAMP}.log"
PULL_DIR="${ROOT_DIR}/results/lesson10_phase2"

mkdir -p "${LOG_DIR}" "${PULL_DIR}"

echo ">>> copying repo to VM" | tee -a "${LOG_PATH}"
"${ROOT_DIR}/scripts/gcp_copy_repo_to_vm.sh" "${PROJECT_ID}" "${ZONE}" "${VM_NAME}" 2>&1 | tee -a "${LOG_PATH}"

REMOTE_CMD='
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
cd ~/cudatraining

find . -name "._*" -delete 2>/dev/null || true

make reduction >/dev/null

mkdir -p ~/lesson10/phase2
cd ~/lesson10/phase2

# Smaller N so that v1 (crippled by atomics) still finishes in reasonable
# time under ncu replay. 4M elements = 16 MB.
N=4194304
ITERS=1         # ncu replays the target kernel automatically; 1 launch is plenty.

echo "=== ncu permission check ==="
# GCP images lock NVIDIA perf counters behind sudo by default
# (NVreg_RestrictProfilingToAdminUsers=1). Easier than rebooting the driver
# with the param flipped — just run ncu via sudo.
NCU="sudo -E env PATH=$PATH ncu"
echo "using: $NCU"

for V in v1 v4; do
  echo
  echo "=== ncu · reduction $V ==="
  # --set detailed : canonical section set (SchedulerStats, WarpStateStats,
  #                  MemoryWorkloadAnalysis, Occupancy, SpeedOfLight, ...).
  #                  Replay adds ~10-30x wall time but that is OK for 1 launch.
  # --launch-skip 20  : skip warmup launches so stats reflect steady-state.
  # --launch-count 1  : capture a single kernel launch.
  # -k "reduce_$V_.*" : regex for the kernel symbol in this version.
  $NCU \
    --set detailed \
    --force-overwrite \
    --launch-skip 20 --launch-count 1 \
    -k "regex:reduce_${V}_" \
    --export "reduction_${V}" \
    ~/cudatraining/bin/reduction \
      --n ${N} --version ${V} --iterations ${ITERS} --warmup 20 \
    || true    # keep going even if the particular version fails

  echo "--- ncu details (human-readable) ---"
  $NCU --import "reduction_${V}.ncu-rep" --page details 2>&1 | \
      grep -E "Elapsed|SM Frequency|Compute \(SM\) Throughput|Memory Throughput|DRAM Throughput|L1/TEX Hit Rate|L2 Hit Rate|Achieved Occupancy|Stall|Warp Cycles|Issued Warp" | \
      head -40 || true
done

ls -la ~/lesson10/phase2/
'

echo ">>> running on VM (ncu replay — expect ~1-2 min)" | tee -a "${LOG_PATH}"
gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --command "bash -lc '${REMOTE_CMD}'" 2>&1 | tee -a "${LOG_PATH}"

echo ">>> pulling .ncu-rep back to ${PULL_DIR}" | tee -a "${LOG_PATH}"
for V in v1 v4; do
  gcloud compute scp \
    --project "${PROJECT_ID}" --zone "${ZONE}" \
    "${VM_NAME}:~/lesson10/phase2/reduction_${V}.ncu-rep" \
    "${PULL_DIR}/reduction_${V}.ncu-rep" 2>&1 | tee -a "${LOG_PATH}" || true
done

echo ">>> done" | tee -a "${LOG_PATH}"
echo "    log         : ${LOG_PATH}"
echo "    ncu reports : ${PULL_DIR}/reduction_{v1,v4}.ncu-rep"
echo
echo "Open in Nsight Compute GUI:"
echo "    ncu-ui ${PULL_DIR}/reduction_v4.ncu-rep &"
echo "  or on Mac:  open -a 'Nsight Compute' ${PULL_DIR}/reduction_v4.ncu-rep"
