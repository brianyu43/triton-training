from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class TimingResult:
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    runs: int


def _synchronize_if_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def time_callable(
    fn: Callable[[], Any],
    *,
    warmup: int = 10,
    iters: int = 50,
) -> TimingResult:
    for _ in range(warmup):
        fn()
    _synchronize_if_cuda()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        _synchronize_if_cuda()
        samples.append((time.perf_counter() - start) * 1000.0)
    return TimingResult(
        median_ms=statistics.median(samples),
        mean_ms=statistics.fmean(samples),
        min_ms=min(samples),
        max_ms=max(samples),
        runs=iters,
    )
