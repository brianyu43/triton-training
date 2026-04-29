from __future__ import annotations

import json
import platform
import sys


def main() -> None:
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": None,
        "triton": None,
        "cuda": None,
        "gpu": None,
        "compute_capability": None,
    }
    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        report["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
            report["compute_capability"] = torch.cuda.get_device_capability(0)
    except Exception as exc:
        report["torch_error"] = repr(exc)

    try:
        import triton

        report["triton"] = triton.__version__
    except Exception as exc:
        report["triton_error"] = repr(exc)

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
