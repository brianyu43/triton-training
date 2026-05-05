# Milestone 5 SwiGLU Standalone Result

Date: 2026-04-29 KST

## Scope

Goal: implement standalone Triton SwiGLU forward/backward before integrating it into the MLP.

Implemented:

- `nanotriton.kernels.swiglu.swiglu_forward`
- `nanotriton.kernels.swiglu.swiglu_backward`
- `TritonSwiGLUFunction`
- CUDA correctness tests against PyTorch autograd
- forward+backward benchmark smoke
- GCP L4 runner for the milestone

The kernel targets only:

```text
swiglu(a, b) = silu(a) * b
```

It does not yet fuse the surrounding linear projections.

## Local

Environment:

- Python: 3.13.12
- torch: not installed
- triton: not installed

Checks:

```text
python3 -m compileall -q nanotriton data scripts benchmarks tests
pytest -q
bash -n scripts/gcp_copy_project_to_vm.sh scripts/gcp_run_milestone1_smoke.sh scripts/gcp_run_milestone2_kernels.sh scripts/gcp_run_milestone3_rmsnorm_backward.sh scripts/gcp_run_milestone4_triton_rmsnorm_model.sh scripts/gcp_run_milestone5_swiglu.sh
git diff --check
```

Result:

```text
ss.. [100%]
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
./scripts/gcp_run_milestone5_swiglu.sh nemo-488500 us-central1-a pxr-chemprop-l4-image-run
```

Correctness result:

```text
................... [100%]
```

This covers:

- forward allclose
- direct backward `da` and `db` comparison against PyTorch autograd
- `TritonSwiGLUFunction` autograd comparison
- fp32/fp16
- shapes including `[16, 128, 512]`
- non-contiguous input

## Forward+Backward Benchmark Smoke

Shape:

```text
[batch, seq, hidden] = [16, 128, 512]
dtype = float16
iters = 100
```

Result:

```json
{
  "speedup": 0.8392920648795363,
  "torch_ms": {
    "median_ms": 0.3728790000003812
  },
  "triton_ms": {
    "median_ms": 0.4442779999997981
  }
}
```

Interpretation: the standalone Triton SwiGLU op is correct but not faster than the PyTorch eager reference for this shape on L4. This is not surprising: standalone elementwise SwiGLU is small, memory-bound, and PyTorch is already efficient here. The useful next step is not to tune this isolated op endlessly, but to fuse it with the MLP path around it.

## Next Step

Move from standalone activation to model integration:

```text
w1(x), w3(x) -> TritonSwiGLU -> w2(...)
```

That will test whether replacing the activation inside the MLP preserves loss curves. Meaningful performance work will likely require a broader fusion target such as projection + activation, not just `silu(a) * b`.

The VM was stopped after validation.
