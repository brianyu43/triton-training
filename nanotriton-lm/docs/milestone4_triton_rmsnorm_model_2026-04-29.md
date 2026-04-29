# Milestone 4 Triton RMSNorm Model Integration Result

Date: 2026-04-29 KST

## Scope

Goal: replace RMSNorm inside the GPT model with `TritonRMSNorm` and verify that model-level training behavior stays aligned with the PyTorch baseline.

Implemented:

- `ModelConfig.norm_impl`
  - `torch`: PyTorch RMSNorm
  - `triton`: Triton RMSNorm forward/backward through autograd
- `configs/tiny_shakespeare_triton_rmsnorm.yaml`
- `scripts/compare_loss_curves.py`
- `scripts/gcp_run_milestone4_triton_rmsnorm_model.sh`
- CUDA model smoke test for `norm_impl="triton"`

This milestone proves that the first Triton Transformer component can be swapped into the training path without changing the architecture or state dict layout.

## Local

Environment:

- Python: 3.13.12
- torch: not installed
- triton: not installed

Checks:

```text
python3 -m compileall -q nanotriton data scripts benchmarks tests
pytest -q
bash -n scripts/gcp_copy_project_to_vm.sh scripts/gcp_run_milestone1_smoke.sh scripts/gcp_run_milestone2_kernels.sh scripts/gcp_run_milestone3_rmsnorm_backward.sh scripts/gcp_run_milestone4_triton_rmsnorm_model.sh
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
./scripts/gcp_run_milestone4_triton_rmsnorm_model.sh nemo-488500 us-central1-a pxr-chemprop-l4-image-run 120
```

CUDA smoke tests:

```text
.................. [100%]
```

## Loss Curve Regression

Both models used:

- same seed: `1337`
- same dataset: Tiny Shakespeare char-level
- same model shape: 4 layers, 4 heads, 128 embedding dim, block size 128
- same max training steps: 120
- same eval interval: 100

PyTorch RMSNorm baseline:

```text
step 0:   train loss 4.2358, val loss 4.2292
step 100: train loss 2.5832, val loss 2.5729
```

Triton RMSNorm model:

```text
step 0:   train loss 4.2358, val loss 4.2292
step 100: train loss 2.5832, val loss 2.5729
```

Detailed comparison:

```json
{
  "final_val_abs_diff": 2.1457672119140625e-05,
  "ref": {
    "final_train": 2.5831820964813232,
    "final_val": 2.5728886127471924,
    "val_delta": -1.6563045978546143
  },
  "triton": {
    "final_train": 2.5832149982452393,
    "final_val": 2.5729100704193115,
    "val_delta": -1.6562817096710205
  }
}
```

The final validation loss difference was `2.15e-05`, far below the configured threshold of `0.25`.

Checkpoint reload and generation also worked:

```text
To bes franek bridcowi,
The ou, bt mad se myobe t e anthand my delatanss ar hthar usq
```

## Interpretation

This is the first end-to-end model integration checkpoint. RMSNorm is no longer just a standalone Triton kernel: it can run inside the GPT training path, backpropagate, checkpoint, reload, and generate.

This does not yet prove end-to-end speedup for the full model. RMSNorm is only one small component. The important result is correctness and training stability under a real loss curve, now with RMSNorm `dx` and `dweight` both staying on the Triton kernel path.

The VM was stopped after validation.
