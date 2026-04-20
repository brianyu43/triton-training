#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:-nemo-488500}"
ZONE="${2:-us-east1-d}"
VM_NAME="${3:-cuda-t4-dev-131020}"

gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --command "bash -lc 'set -euo pipefail; cd ~/cudatraining; find . -name \"._*\" -delete; mkdir -p results bin; if ! command -v make >/dev/null 2>&1; then sudo apt-get update && sudo apt-get install -y build-essential; fi; ./scripts/check_cuda_env.sh | tee results/check_cuda_env.txt; /usr/local/cuda/bin/nvcc -O3 -std=c++17 -lineinfo -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 src/vector_add.cu -o bin/vector_add; { ./bin/vector_add --n 67108864 --block-size 256 --iterations 100; ./bin/vector_add --n 67108864 --block-size 128 --iterations 100 --csv; ./bin/vector_add --n 67108864 --block-size 512 --iterations 100 --csv; } | tee results/vector_add_t4.txt'"
