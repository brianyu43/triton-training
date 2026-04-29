# GPU Kernel Projects

This workspace contains two separate tracks:

## CUDA/Triton Lessons

`cuda-triton-lessons/` is the archived lesson series from Lesson 01 through
Lesson 12. It contains the CUDA kernels, Triton kernels, benchmarks, scripts,
handoff notes, and blog drafts built during the training sequence.

Start there when revisiting the step-by-step learning record:

```bash
cd cuda-triton-lessons
```

## NanoTriton-LM

`nanotriton-lm/` is a new independent project: a tiny GPT-style training stack
where core Transformer operations are replaced with custom Triton kernels and
validated through tests, benchmarks, and profiler reports.

The project plan lives at:

```text
nanotriton-lm/PROJECT_PLAN.md
```

The two folders should evolve independently: lessons stay as a record of the
training path, while NanoTriton-LM becomes the portfolio project.
