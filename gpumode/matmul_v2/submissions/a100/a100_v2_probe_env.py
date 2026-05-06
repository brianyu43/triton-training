#!POPCORN leaderboard matmul_v2
#!POPCORN gpus A100

import ctypes
import ctypes.util
import os
import shutil
import sys

import torch
from task import input_t, output_t


_PROBED = False


def _try_load(names):
    for name in names:
        if not name:
            continue
        try:
            lib = ctypes.CDLL(name)
            return name, "ok", str(lib._handle)
        except OSError as exc:
            last = str(exc).splitlines()[0]
    return ",".join(n for n in names if n), "fail", last if names else "not-found"


def _probe_once():
    global _PROBED
    if _PROBED:
        return
    _PROBED = True

    candidates = {
        "cublasLt": [
            ctypes.util.find_library("cublasLt"),
            "libcublasLt.so",
            "libcublasLt.so.12",
            "/usr/local/cuda/lib64/libcublasLt.so",
        ],
        "cublas": [
            ctypes.util.find_library("cublas"),
            "libcublas.so",
            "libcublas.so.12",
            "/usr/local/cuda/lib64/libcublas.so",
        ],
        "cudart": [
            ctypes.util.find_library("cudart"),
            "libcudart.so",
            "libcudart.so.12",
            "/usr/local/cuda/lib64/libcudart.so",
        ],
    }

    print("PROBE: python", sys.version.split()[0], file=sys.stderr)
    print("PROBE: torch", torch.__version__, file=sys.stderr)
    print("PROBE: cuda_available", torch.cuda.is_available(), file=sys.stderr)
    if torch.cuda.is_available():
        print("PROBE: device", torch.cuda.get_device_name(), file=sys.stderr)
        print("PROBE: capability", torch.cuda.get_device_capability(), file=sys.stderr)
    print("PROBE: LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH", ""), file=sys.stderr)
    for tool in ("nvcc", "c++", "g++", "ninja"):
        print(f"PROBE: tool {tool}", shutil.which(tool), file=sys.stderr)

    try:
        import torch.utils.cpp_extension as cpp_extension

        print("PROBE: cpp_extension ok", cpp_extension.CUDA_HOME, file=sys.stderr)
    except Exception as exc:
        print("PROBE: cpp_extension fail", type(exc).__name__, str(exc), file=sys.stderr)

    for key, names in candidates.items():
        loaded, status, detail = _try_load(names)
        print(f"PROBE: load {key} {status} {loaded} {detail}", file=sys.stderr)


def custom_kernel(data: input_t) -> output_t:
    _probe_once()
    a, b, c = data
    torch.mm(a, b, out=c)
    return c
