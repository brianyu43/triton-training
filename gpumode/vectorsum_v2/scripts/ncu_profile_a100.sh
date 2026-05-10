#!/usr/bin/env bash
set -euo pipefail

INSTANCE="${INSTANCE:-cuda-a100-dev-matmul-v2}"
ZONE="${ZONE:-us-central1-a}"
THREADS="${THREADS:-256}"
BLOCKS_PER_SM="${BLOCKS_PER_SM:-12}"
SIZE="${SIZE:-52428800}"
SEED="${SEED:-12345}"
REPS="${REPS:-1}"
SET_NAME="${SET_NAME:-basic}"
KERNEL_FILTER="${KERNEL_FILTER:-regex:.*(read_partial_tile4_kernel|atomic_tile4_kernel).*}"
NCU_BIN="${NCU_BIN:-ncu}"
NCU_SUDO="${NCU_SUDO:-0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/gpumode/vectorsum_v2/logs"
LOCAL_LOG="$LOG_DIR/ncu_profile_a100_${STAMP}.out"
LOCAL_REPORT="$LOG_DIR/ncu_profile_a100_${STAMP}"

REMOTE_ROOT="/home/xavier"
REMOTE_PROJ="$REMOTE_ROOT/cudatraining/gpumode/vectorsum_v2"
REMOTE_SCRIPT="$REMOTE_PROJ/scripts/roofline_probe_a100.py"
REMOTE_REPORT="$REMOTE_PROJ/logs/ncu_profile_a100_${STAMP}"

mkdir -p "$LOG_DIR"

STATUS="$(gcloud compute instances describe "$INSTANCE" --zone "$ZONE" --format='value(status)')"
if [[ "$STATUS" != "RUNNING" ]]; then
  echo "VM $INSTANCE is $STATUS. Start it first with gpumode/vectorsum_v2/scripts/gcp_start_a100.sh" >&2
  exit 3
fi

gcloud compute ssh "$INSTANCE" --zone "$ZONE" --command \
  "mkdir -p '$REMOTE_PROJ/scripts' '$REMOTE_PROJ/logs'"
gcloud compute scp "$ROOT_DIR/gpumode/vectorsum_v2/scripts/roofline_probe_a100.py" \
  "$INSTANCE:$REMOTE_SCRIPT" --zone "$ZONE"

set +e
gcloud compute ssh "$INSTANCE" --zone "$ZONE" --command "
set -euo pipefail
export PATH=\"\$HOME/.local/bin:/usr/local/cuda/bin:/usr/local/cuda-12.9/bin:\$PATH\"
cd '$REMOTE_ROOT/cudatraining'
NCU_BIN='$NCU_BIN'
NCU_SUDO='$NCU_SUDO'
if ! command -v \"\$NCU_BIN\" >/dev/null 2>&1; then
  echo 'ncu not found on remote PATH' >&2
  exit 127
fi
run_ncu() {
  if [ \"\$NCU_SUDO\" = '1' ]; then
    sudo -E env \
      HOME=\"\$HOME\" \
      PYTHONPATH=\"\$HOME/.local/lib/python3.10/site-packages:\${PYTHONPATH:-}\" \
      PATH=\"\$PATH\" \
      \"\$NCU_BIN\" \"\$@\"
  else
    \"\$NCU_BIN\" \"\$@\"
  fi
}
run_ncu --version
run_ncu \
  --set '$SET_NAME' \
  --target-processes all \
  --kernel-name '$KERNEL_FILTER' \
  --force-overwrite \
  --export '$REMOTE_REPORT' \
  python3 '$REMOTE_SCRIPT' \
    --size '$SIZE' \
    --seed '$SEED' \
    --threads '$THREADS' \
    --blocks-per-sm '$BLOCKS_PER_SM' \
    --reps '$REPS' \
    --warmups 0 \
    --only read_partial_tile4 v11_atomic_tile4
" 2>&1 | tee "$LOCAL_LOG"
status="${PIPESTATUS[0]}"
set -e

if [[ "$status" -eq 0 ]]; then
  gcloud compute scp "$INSTANCE:$REMOTE_REPORT.ncu-rep" "$LOCAL_REPORT.ncu-rep" --zone "$ZONE" >/dev/null 2>&1 || true
  echo "local-log: $LOCAL_LOG"
  if [[ -f "$LOCAL_REPORT.ncu-rep" ]]; then
    echo "local-report: $LOCAL_REPORT.ncu-rep"
  fi
fi

exit "$status"
