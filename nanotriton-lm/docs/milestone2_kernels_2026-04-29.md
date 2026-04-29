# Milestone 2 Kernel Smoke Result

Date: 2026-04-29 KST

## Scope

Goal: establish the first Triton kernel correctness loop before replacing model modules.

Implemented:

- `nanotriton.kernels.vector_add.vector_add`
- `nanotriton.kernels.rmsnorm.rmsnorm_forward`
- CUDA correctness tests against PyTorch
- RMSNorm forward benchmark smoke
- GCP L4 runner for the kernel milestone

## Local

Environment:

- Python: 3.13.12
- torch: not installed
- triton: not installed

Checks:

```text
python3 -m compileall -q nanotriton data scripts benchmarks tests
pytest -q
bash -n scripts/gcp_copy_project_to_vm.sh scripts/gcp_run_milestone1_smoke.sh scripts/gcp_run_milestone2_kernels.sh
git diff --check
```

Result:

```text
s.. [100%]
```

CUDA tests are skipped locally because torch/triton are not installed.

## L4 VM

VM:

- GCP project: `nemo-488500`
- VM: `pxr-chemprop-l4-image-run`
- Zone: `us-central1-a`
- GPU: NVIDIA L4

GPU environment:

```json
{
  "compute_capability": [8, 9],
  "cuda": "13.0",
  "cuda_available": true,
  "gpu": "NVIDIA L4",
  "python": "3.10.12",
  "torch": "2.11.0+cu130",
  "triton": "3.6.0"
}
```

Command:

```bash
./scripts/gcp_run_milestone2_kernels.sh nemo-488500 us-central1-a pxr-chemprop-l4-image-run
```

Correctness result:

```text
.............. [100%]
```

This covers:

- vector add: fp32/fp16, contiguous and non-contiguous inputs
- RMSNorm forward: fp32/fp16, hidden sizes 16/65/128, contiguous and non-contiguous inputs

## RMSNorm Benchmark Smoke

Shape:

```text
[batch, seq, hidden] = [16, 128, 128]
dtype = float16
iters = 100
```

Result:

```json
{
  "speedup": 2.697722476612371,
  "torch_ms": {
    "median_ms": 0.15339250000323545
  },
  "triton_ms": {
    "median_ms": 0.05685999999371916
  }
}
```

Interpretation: this is only a first smoke benchmark, not yet a tuned performance claim. It proves the benchmark path works and that the simple Triton RMSNorm forward is already meaningfully faster than the PyTorch eager reference for this small training shape on L4.

The VM was stopped after validation.
