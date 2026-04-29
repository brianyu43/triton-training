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

Current status: planning/scaffolding.
