#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:-nemo-488500}"
ZONE="${2:-us-east1-d}"
VM_NAME="${3:-cuda-t4-dev-131020}"
STAMP="${4:-$(date +%Y%m%d-%H%M%S)}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
LOG_DIR="${ROOT_DIR}/results"
LOG_PATH="${LOG_DIR}/gcp-run-${STAMP}.log"

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

echo "Visible SSH session: install compiler, compile, benchmark"
echo "Log: ${LOG_PATH}"

"${ROOT_DIR}/scripts/gcp_visible_ssh_run.sh" "${PROJECT_ID}" "${ZONE}" "${VM_NAME}" 2>&1 | tee "${LOG_PATH}"
