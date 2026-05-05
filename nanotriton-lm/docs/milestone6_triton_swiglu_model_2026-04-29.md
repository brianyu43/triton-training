# Milestone 6 Triton SwiGLU Model Integration Result

Date: 2026-04-29 KST

## Scope

Goal: replace the SwiGLU activation inside the GPT MLP with the standalone
Triton SwiGLU autograd function and verify that model-level training behavior
stays aligned with the PyTorch baseline.

Implemented:

- `ModelConfig.mlp_impl`
  - `torch`: PyTorch `F.silu(a) * b`
  - `triton_swiglu`: Triton SwiGLU forward/backward through autograd
- `configs/tiny_shakespeare_triton_swiglu.yaml`
- `scripts/gcp_run_milestone6_triton_swiglu_model.sh`
- CUDA model smoke test for `mlp_impl="triton_swiglu"`

This milestone keeps the surrounding linear layers in PyTorch:

```text
w1(x), w3(x) -> TritonSwiGLU -> w2(...)
```

It does not yet fuse projection, activation, and output projection into one
larger MLP kernel.

## Local

Environment:

- Python: 3.13.12
- torch: not installed
- triton: not installed

Checks:

```text
bash -n scripts/gcp_copy_project_to_vm.sh scripts/gcp_run_milestone1_smoke.sh scripts/gcp_run_milestone2_kernels.sh scripts/gcp_run_milestone3_rmsnorm_backward.sh scripts/gcp_run_milestone4_triton_rmsnorm_model.sh scripts/gcp_run_milestone5_swiglu.sh scripts/gcp_run_milestone6_triton_swiglu_model.sh
python3 -m compileall -q nanotriton data scripts benchmarks tests
pytest -q
git diff --check
```

Result:

```text
sss.. [100%]
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
./scripts/gcp_run_milestone6_triton_swiglu_model.sh nemo-488500 us-central1-a pxr-chemprop-l4-image-run 120
```

CUDA smoke tests:

```text
...................... [100%]
```

This covers the standalone SwiGLU CUDA tests plus a GPT forward/backward pass
with `mlp_impl="triton_swiglu"`.

## Loss Curve Regression

Both models used:

- same seed: `1337`
- same dataset: Tiny Shakespeare char-level
- same model shape: 4 layers, 4 heads, 128 embedding dim, block size 128
- same max training steps: 120
- same eval interval: 100

PyTorch SwiGLU baseline:

```text
step 0:   train loss 4.2358, val loss 4.2292
step 100: train loss 2.5832, val loss 2.5729
```

Triton SwiGLU model:

```text
step 0:   train loss 4.2358, val loss 4.2292
step 100: train loss 2.5832, val loss 2.5729
```

Detailed comparison:

```json
{
  "final_val_abs_diff": 1.7881393432617188e-05,
  "ref": {
    "final_train": 2.5831820964813232,
    "final_val": 2.5728886127471924,
    "val_delta": -1.6563045978546143
  },
  "triton": {
    "final_train": 2.5831940174102783,
    "final_val": 2.572906494140625,
    "val_delta": -1.6562881469726562
  }
}
```

The final validation loss difference was `1.79e-05`, far below the configured
threshold of `0.25`.

Checkpoint reload and generation also worked:

```text
To bes franek bridcowi,
The ou, bt mad se myobe t e anthand my delatanss ar hthar usq
```

## Interpretation

The standalone SwiGLU kernel now participates in real GPT training and preserves
the loss curve. That is a meaningful correctness checkpoint.

It is not a speedup checkpoint yet. Milestone 5 showed that the isolated
`silu(a) * b` Triton op was slower than PyTorch eager on L4, and Milestone 6
does not change the performance story by itself. The next performance-relevant
step is a broader MLP fusion target or a profiler pass that identifies the
dominant remaining overhead before writing more kernels.

The VM was stopped after validation.
