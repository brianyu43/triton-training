#!/usr/bin/env bash
set -euo pipefail

echo "== nvidia-smi =="
nvidia-smi

echo
echo "== nvcc --version =="
nvcc --version

echo
echo "== GPU compute capability overview =="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

