# Third-party kernel attribution

The file `unified_attention.py` in this directory is a minimal-dependency
extraction of vLLM's `vllm/v1/attention/ops/triton_unified_attention.py`,
pinned against upstream SHA at the time of extraction (Lesson 13, 2026-04-22).

Upstream license: Apache License 2.0. Copyright contributors to the vLLM
project. Named upstream authors (from the source file header):

- Burkhard Ringlein <ngl@zurich.ibm.com>
- Jan van Lunteren <jvl@zurich.ibm.com>
- Chih-Chieh Yang <chih.chieh.yang@ibm.com>
- Thomas Parnell <tpa@zurich.ibm.com>

## What was changed vs upstream

The kernel bodies (`kernel_unified_attention_2d`, `kernel_unified_attention_3d`,
`reduce_segments`, `_prepare_kv_tile`, `find_seq_idx`, `apply_softcap`,
`cdiv_fn`) are verbatim. Only the module-level imports and a handful of
downstream-specific helpers were replaced to make the file importable outside
vLLM's package tree:

| Upstream dependency                     | Local replacement                 |
| --------------------------------------- | --------------------------------- |
| `import vllm.envs as envs`              | removed                           |
| `envs.VLLM_BATCH_INVARIANT`             | module constant `False`           |
| `from vllm.logger import init_logger`   | removed                           |
| `logger = init_logger(__name__)`        | removed                           |
| `from vllm.platforms import ...`        | `torch.finfo(torch.float8_e4m3fn)`|
| `from vllm.triton_utils import tl, triton` | standard `triton`, `triton.language as tl` |
| `from vllm.v1.kv_cache_interface import KVQuantMode` | local `IntEnum` with matching values |

All three launch sites (`unified_attention(...)`) retain their full vLLM
signature. The `is_batch_invariant` branch is preserved as a dead branch
guarded by the constant `False`, so the dispatch logic on line 1157-1166
of the upstream file remains byte-identical.

## Why a copy rather than `pip install vllm`

- We intend to probe the `seq_threshold_3D` dispatch heuristic on an L4 GPU,
  which requires modifying the call site (not the kernel). A copy lets us
  own the call site without depending on vLLM internals being stable.
- vLLM installation pulls a full CUDA toolchain / torch version pin that
  conflicts with this repo's lighter env.
- The upstream file is self-contained (no cross-module Triton helpers) once
  the 5 imports above are stubbed — the cost of a copy is low.

## Staying in sync

Upstream is actively developed. If upstream changes the kernel or the
dispatch arithmetic, the benchmark results in
`docs/vllm_audit_02_heuristic_b_kernel_bench.md` are scoped to the pinned
SHA, not to "vLLM main". Re-extract and re-run the bench before claiming
parity with a newer version.
