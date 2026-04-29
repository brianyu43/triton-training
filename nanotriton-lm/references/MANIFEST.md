# Reference Manifest

Fetched reference repositories are local study material and are not part of NanoTriton-LM implementation code.

| name | source | pinned commit | license | purpose |
|---|---|---|---|---|
| nanogpt | https://github.com/karpathy/nanoGPT | `3adf61e154c3fe3fca428ad6bc3818b27a3b8291` | MIT | minimal GPT training loop, config, and generation reference |
| triton | https://github.com/triton-lang/triton | `0f5f46ef80b90488f3dd9f64737ad79c3a6cafe6` | MIT-style | Triton tutorial and kernel implementation patterns |
| flash-attention | https://github.com/Dao-AILab/flash-attention | `ba59def94cd7a0c12e2a8c673b0a4655be67c5c4` | BSD-3-Clause | attention forward/backward math and benchmarking reference |

Reference directories under `references/*/` are intentionally gitignored; rerun `python scripts/fetch_references.py --name all` to reconstruct them.