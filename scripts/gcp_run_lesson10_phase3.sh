#!/usr/bin/env bash
set -euo pipefail

# Lesson 10 · Phase 3 : ncu diff for lesson 09 (ours Triton MHA FA vs SDPA/cuDNN FA-2).
#
# Fixed shape: B=1 H=32 N=2048 d=128 causal fp16 — a LLaMA-7B mid-range shape
# where Phase 3 of lesson 09 measured ours = 0.78× SDPA.
#
# We expect to see, for SDPA vs ours:
#   - higher tensor-core utilization (sm__pipe_tensor_*)
#   - fewer total cycles
#   - fewer scheduler stalls waiting on memory
# i.e. the ~20% gap should be visible as either tensor path usage OR stall mix.

PROJECT_ID="${1:-nemo-488500}"
ZONE="${2:-us-west1-b}"
VM_NAME="${3:-cuda-l4-dev-lesson10}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${ROOT_DIR}/results"
LOG_PATH="${LOG_DIR}/lesson10-phase3-${STAMP}.log"
PULL_DIR="${ROOT_DIR}/results/lesson10_phase3"

mkdir -p "${LOG_DIR}" "${PULL_DIR}"

echo ">>> copying repo to VM" | tee -a "${LOG_PATH}"
"${ROOT_DIR}/scripts/gcp_copy_repo_to_vm.sh" "${PROJECT_ID}" "${ZONE}" "${VM_NAME}" 2>&1 | tee -a "${LOG_PATH}"

REMOTE_CMD='
set -euo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$PATH
cd ~/cudatraining

find . -name "._*" -delete 2>/dev/null || true

# Sanity check env (torch + triton already installed in Phase 3 bootstrap).
python3 -c "import torch, triton; print(\"torch\", torch.__version__, \"triton\", triton.__version__, \"cap=sm_\"+str(torch.cuda.get_device_capability(0)[0])+str(torch.cuda.get_device_capability(0)[1]))"

mkdir -p ~/lesson10/phase3
cd ~/lesson10/phase3

B=1 H=32 N=2048 D=128

echo
echo "=== discovery: list kernel names for each mode ==="
for MODE in ours sdpa; do
  echo "--- $MODE kernels ---"
  python3 ~/cudatraining/triton_kernels/bench/lesson10_phase3_profile.py \
    --mode ${MODE} --B ${B} --H ${H} --N ${N} --d ${D} \
    --warmup 20 --iterations 1 --list-kernels 2>&1 | tail -30
done

NCU="sudo -E env PATH=$PATH ncu"
echo
echo "ncu via: $NCU"
$NCU --version | head -1

# Kernel filters: ours = triton-generated symbol; sdpa = broad attention regex.
OURS_REGEX="flash_attention_mha"
SDPA_REGEX="flash|fmha|attention|sdpa|cudnn"

for MODE in ours sdpa; do
  REGEX="${OURS_REGEX}"
  if [ "$MODE" = "sdpa" ]; then REGEX="${SDPA_REGEX}"; fi

  echo
  echo "=== ncu · $MODE (regex: ${REGEX}) ==="
  # --set detailed : SOL + MemoryWorkload + Occupancy + LaunchStats + SchedulerStats
  # --launch-skip 20 --launch-count 1 : after python warmup=20, capture exactly
  #                                      ONE matching kernel launch.
  $NCU \
    --set detailed \
    --force-overwrite \
    --launch-skip 20 --launch-count 1 \
    -k "regex:${REGEX}" \
    --export "fa_${MODE}" \
    python3 ~/cudatraining/triton_kernels/bench/lesson10_phase3_profile.py \
      --mode ${MODE} --B ${B} --H ${H} --N ${N} --d ${D} \
      --warmup 20 --iterations 1 \
    || true

  ls -la "fa_${MODE}.ncu-rep" 2>/dev/null || echo "(no report produced)"

  echo "--- details summary ($MODE) ---"
  $NCU --import "fa_${MODE}.ncu-rep" --page details 2>&1 | \
    grep -E "Kernel Name|Elapsed|SM Frequency|Compute \(SM\) Throughput|Memory Throughput|DRAM Throughput|L1/TEX Hit Rate|L2 Hit Rate|Achieved Occupancy|Registers Per Thread|Block Size|Grid Size|Tensor|Pipe" | \
    head -40 || true
done

# Extract the stall breakdown just like Phase 2.
cat > extract_stalls.py <<PYEOF
import csv, sys, json
rows = list(csv.reader(sys.stdin))
if len(rows) < 3:
    print(json.dumps({"total": 0, "reasons": []}))
    sys.exit(0)
headers, values = rows[0], rows[2]
pairs = [(h, v) for h, v in zip(headers, values)
         if "pcsamp_warps_issue_stalled" in h and "not_issued" not in h]
parsed = [(h.replace("smsp__pcsamp_warps_issue_stalled_", ""), float(v))
          for h, v in pairs if v not in ("", "-", "N/A")]
total = sum(v for _, v in parsed)
parsed.sort(key=lambda x: -x[1])
print(json.dumps({
    "kernel": rows[2][4] if len(rows[2]) > 4 else "",
    "total": total,
    "reasons": [(n, v, 100*v/total if total else 0) for n, v in parsed],
}, indent=2))
PYEOF

for MODE in ours sdpa; do
  echo
  echo "=== stall reasons · $MODE ==="
  $NCU --import "fa_${MODE}.ncu-rep" --csv --page raw 2>/dev/null \
    | python3 extract_stalls.py | tee "stall_${MODE}.json"
done

ls -la ~/lesson10/phase3/
'

echo ">>> running on VM (ncu replay — expect ~2-3 min)" | tee -a "${LOG_PATH}"
gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --command "bash -lc '${REMOTE_CMD}'" 2>&1 | tee -a "${LOG_PATH}"

echo ">>> pulling .ncu-rep + stall JSON back to ${PULL_DIR}" | tee -a "${LOG_PATH}"
for MODE in ours sdpa; do
  for EXT in ncu-rep; do
    gcloud compute scp \
      --project "${PROJECT_ID}" --zone "${ZONE}" \
      "${VM_NAME}:~/lesson10/phase3/fa_${MODE}.${EXT}" \
      "${PULL_DIR}/fa_${MODE}.${EXT}" 2>&1 | tee -a "${LOG_PATH}" || true
  done
  gcloud compute scp \
    --project "${PROJECT_ID}" --zone "${ZONE}" \
    "${VM_NAME}:~/lesson10/phase3/stall_${MODE}.json" \
    "${PULL_DIR}/stall_${MODE}.json" 2>&1 | tee -a "${LOG_PATH}" || true
done

echo ">>> done" | tee -a "${LOG_PATH}"
echo "    log         : ${LOG_PATH}"
echo "    ncu reports : ${PULL_DIR}/fa_{ours,sdpa}.ncu-rep"
echo "    stall JSON  : ${PULL_DIR}/stall_{ours,sdpa}.json"
echo
echo "Open in Nsight Compute GUI:"
echo "    ncu-ui ${PULL_DIR}/fa_sdpa.ncu-rep &"
echo "  or on Mac:  open -a 'Nsight Compute' ${PULL_DIR}/fa_sdpa.ncu-rep"
