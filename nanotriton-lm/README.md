# NanoTriton-LM

NanoTriton-LM is an independent long-form project for building and training a
tiny GPT-style language model while replacing core Transformer operations with
custom Triton kernels.

The goal is not to train a strong language model. The goal is to demonstrate a
full engineering loop:

- PyTorch reference model
- Triton kernel implementation
- forward and backward correctness checks
- microbenchmarks
- end-to-end training benchmarks
- profiler-based performance analysis

The initial plan is in [PROJECT_PLAN.md](PROJECT_PLAN.md). This is not a lesson
log; it is meant to grow into its own portfolio repository inside this
workspace.

## Milestone 1: Reference Training Stack

Local setup:

```bash
cd nanotriton-lm
python3 -m pip install -e '.[dev]'
python3 scripts/fetch_references.py --name all
python3 data/shakespeare_char/prepare.py
pytest -q
```

GPU smoke test:

```bash
python3 scripts/env_report.py
python3 -m nanotriton.train --config configs/tiny_shakespeare_ref.yaml --max-iters 20
python3 -m nanotriton.generate \
  --ckpt out/tiny_shakespeare_ref/checkpoint.pt \
  --prompt "To be" \
  --max-new-tokens 80
```

VM notes are in [docs/vm_selection.md](docs/vm_selection.md). Smoke results are
in [docs/milestone1_smoke_2026-04-29.md](docs/milestone1_smoke_2026-04-29.md).
Current decision: A100 on-demand is blocked by quota on both checked AWS and GCP
paths, so the repeatable fallback is an existing GCP L4 VM:

```bash
gcloud compute instances start pxr-chemprop-l4-image-run \
  --project nemo-488500 \
  --zone us-central1-a

./scripts/gcp_run_milestone1_smoke.sh nemo-488500 us-central1-a pxr-chemprop-l4-image-run 120
```

## Milestone 2: First Triton Kernels

Kernel smoke results are in
[docs/milestone2_kernels_2026-04-29.md](docs/milestone2_kernels_2026-04-29.md).

```bash
./scripts/gcp_run_milestone2_kernels.sh nemo-488500 us-central1-a pxr-chemprop-l4-image-run
```

Current status: PyTorch reference baseline plus first Triton vector add and
RMSNorm forward correctness loop.

## Reinforcement Learning Track

The proposed RL training-system track is in
[docs/rl_training_system_plan.md](docs/rl_training_system_plan.md). It should
start after the RMSNorm backward/autograd and model integration path is stable.

## Milestone 3: RMSNorm Backward

RMSNorm backward and autograd smoke results are in
[docs/milestone3_rmsnorm_backward_2026-04-29.md](docs/milestone3_rmsnorm_backward_2026-04-29.md).

```bash
./scripts/gcp_run_milestone3_rmsnorm_backward.sh nemo-488500 us-central1-a pxr-chemprop-l4-image-run
```

Current status: RMSNorm now has Triton forward/backward kernels, a fully Triton
`dweight` reduction path, an autograd function, and an `nn.Module` wrapper.

## Milestone 4: Triton RMSNorm Model Integration

Model-level loss regression results are in
[docs/milestone4_triton_rmsnorm_model_2026-04-29.md](docs/milestone4_triton_rmsnorm_model_2026-04-29.md).

```bash
./scripts/gcp_run_milestone4_triton_rmsnorm_model.sh nemo-488500 us-central1-a pxr-chemprop-l4-image-run 120
```

Current status: GPT training can run with PyTorch RMSNorm or Triton RMSNorm via
`ModelConfig.norm_impl`, and the 120-step Tiny Shakespeare loss curves match.

## Milestone 5: Standalone SwiGLU

Standalone SwiGLU forward/backward results are in
[docs/milestone5_swiglu_2026-04-29.md](docs/milestone5_swiglu_2026-04-29.md).

```bash
./scripts/gcp_run_milestone5_swiglu.sh nemo-488500 us-central1-a pxr-chemprop-l4-image-run
```

Current status: SwiGLU standalone correctness passes, but the isolated Triton
op is slower than PyTorch eager on L4. The next useful target is MLP integration
and then broader fusion.

## Milestone 6: Triton SwiGLU Model Integration

Model-level loss regression results are in
[docs/milestone6_triton_swiglu_model_2026-04-29.md](docs/milestone6_triton_swiglu_model_2026-04-29.md).

```bash
./scripts/gcp_run_milestone6_triton_swiglu_model.sh nemo-488500 us-central1-a pxr-chemprop-l4-image-run 120
```

Current status: GPT training can run with PyTorch SwiGLU or standalone Triton
SwiGLU via `ModelConfig.mlp_impl`. The 120-step Tiny Shakespeare loss curves
match, but this is still a correctness milestone rather than an end-to-end
speedup milestone.
