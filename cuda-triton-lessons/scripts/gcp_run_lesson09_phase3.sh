#!/usr/bin/env bash
set -euo pipefail

# Lesson 09 · Phase 3 : 3-way MHA Flash Attention speed benchmark on L4.

PROJECT_ID="${1:-nemo-488500}"
ZONE="${2:-us-east4-c}"
VM_NAME="${3:-cuda-l4-dev-lesson09}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${ROOT_DIR}/results"
LOG_PATH="${LOG_DIR}/lesson09-phase3-${STAMP}.log"

mkdir -p "${LOG_DIR}"

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

echo "+++ Correctness pre-check (fast)"
$PY -c "
import torch, torch.nn.functional as F
import sys
sys.path.insert(0, \".\")
from triton_kernels.flash_attention_mha import triton_flash_attention_mha
for ic in (False, True):
    q=torch.randn(1,32,1024,128,device=\"cuda\",dtype=torch.float16)
    k=torch.randn_like(q); v=torch.randn_like(q)
    ref=F.scaled_dot_product_attention(q.float(),k.float(),v.float(),is_causal=ic)
    ours=triton_flash_attention_mha(q,k,v,is_causal=ic).float()
    e=(ours-ref).abs().max().item()/ref.abs().max().item()
    assert e < 5e-2, (ic, e)
    print(f\"  causal={ic} rel_err={e:.2e}  ok\")
"

echo "+++ Phase 3 speed bench"
$PY triton_kernels/bench/bench_flash_attention_mha_speed.py
'

echo ">>> running on VM" | tee -a "${LOG_PATH}"
gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --command "bash -lc '${REMOTE_CMD}'" 2>&1 | tee -a "${LOG_PATH}"

echo ">>> done" | tee -a "${LOG_PATH}"
echo "    log : ${LOG_PATH}"
