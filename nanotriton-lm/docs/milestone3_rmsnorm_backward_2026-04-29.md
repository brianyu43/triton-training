# Milestone 3 RMSNorm Backward Result

Date: 2026-04-29 KST

## Scope

Goal: make RMSNorm replaceable as a PyTorch autograd component, not just a forward-only kernel.

Implemented:

- Triton RMSNorm backward kernel
- `TritonRMSNormFunction`
- `TritonRMSNorm` module wrapper
- gradient correctness tests against PyTorch autograd
- forward+backward benchmark smoke
- GCP L4 runner for the milestone

Current implementation detail:

- `dx` is computed in Triton.
- `dweight` is produced as row-wise fp32 partials from Triton and reduced with PyTorch `sum(dim=0)`.

This keeps the milestone stable while leaving a clear future optimization target: a fully Triton `dweight` reduction.

## Backward Formula

For each row:

```text
rstd = 1 / sqrt(mean(x^2) + eps)
y_i = x_i * rstd * weight_i

dot = sum_j(grad_y_j * weight_j * x_j)
dx_i = grad_y_i * weight_i * rstd - x_i * dot * rstd^3 / N
dweight_i = sum_rows(grad_y_i * x_i * rstd)
```

## Local

Environment:

- Python: 3.13.12
- torch: not installed
- triton: not installed

Checks:

```text
python3 -m compileall -q nanotriton data scripts benchmarks tests
pytest -q
bash -n scripts/gcp_copy_project_to_vm.sh scripts/gcp_run_milestone1_smoke.sh scripts/gcp_run_milestone2_kernels.sh scripts/gcp_run_milestone3_rmsnorm_backward.sh
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
./scripts/gcp_run_milestone3_rmsnorm_backward.sh nemo-488500 us-central1-a pxr-chemprop-l4-image-run
```

Correctness result:

```text
................ [100%]
```

This covers:

- forward allclose
- direct backward `dx` and `dweight` comparison against PyTorch autograd
- `TritonRMSNormFunction` autograd comparison
- `TritonRMSNorm` module wrapper comparison
- fp32/fp16
- hidden sizes 16/65/128
- contiguous and non-contiguous input cases

## Forward+Backward Benchmark Smoke

Shape:

```text
[batch, seq, hidden] = [16, 128, 128]
dtype = float16
iters = 100
```

Result:

```json
{
  "speedup": 1.3937598075148616,
  "torch_ms": {
    "median_ms": 0.7536344999969913
  },
  "triton_ms": {
    "median_ms": 0.5407204999983151
  }
}
```

Interpretation: this is a first forward+backward smoke benchmark, not a final performance claim. The result is already faster than the PyTorch eager reference for this training shape on L4, while still leaving `dweight` reduction as an optimization target.

The VM was stopped after validation.
