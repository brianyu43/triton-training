# Milestone 1 Smoke Result

Date: 2026-04-29 KST

## Local

Environment:

- Python: 3.13.12
- torch: not installed
- triton: not installed

Checks:

```text
python3 -m compileall -q nanotriton data scripts benchmarks tests
python3 scripts/env_report.py
pytest -q
```

Result:

```text
s.. [100%]
```

The model test is skipped locally because torch is not installed. Config and tokenizer sanity tests pass.

## References

Fetched and pinned:

```text
nanogpt          3adf61e154c3fe3fca428ad6bc3818b27a3b8291
triton           0f5f46ef80b90488f3dd9f64737ad79c3a6cafe6
flash-attention  ba59def94cd7a0c12e2a8c673b0a4655be67c5c4
```

Reference directories are ignored by git and reproducible via:

```bash
python3 scripts/fetch_references.py --name all
```

## Dataset

Tiny Shakespeare char-level cache:

```text
chars: 1115394
vocab_size: 65
train tokens: 1003854
val tokens: 111540
```

Generated cache files are ignored by git.

## VM Selection

A100 on-demand was checked first.

- AWS `ap-northeast-2`: `p4d.24xlarge` exists in `ap-northeast-2b` and `ap-northeast-2d`, but `Running On-Demand P instances = 0`.
- GCP: A2/A100 zones exist, but checked regions have `NVIDIA_A100_GPUS = 0` and `NVIDIA_A100_80GB_GPUS = 0`.

L4 fallback:

- `cuda-l4-dev-lesson09` in `us-east4-c` failed to start due to L4 stockout.
- `pxr-chemprop-l4-image-run` in `us-central1-a` started successfully.
- VM was stopped after validation.

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

## Training Smoke

Command:

```bash
python3 -m nanotriton.train \
  --config configs/tiny_shakespeare_ref.yaml \
  --max-iters 120 \
  --out-dir out/tiny_shakespeare_ref_120
```

Observed loss:

```text
step 0:   train loss 4.2358, val loss 4.2292, lr 1.20e-05
step 100: train loss 2.5832, val loss 2.5729, lr 5.96e-04
```

Checkpoint reload and generation also worked:

```text
To bes franek bridcowi,
The ou, bt mad se myobe t e anthand my delatanss ar hthar usq
```
