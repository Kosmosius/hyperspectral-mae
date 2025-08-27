from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


def linear_ramp(start: int, end: int, v0: float, v1: float) -> Callable[[int], float]:
    """
    Piecewise-linear ramp from v0 at epoch=start to v1 at epoch=end. Clamped outside.
    """
    start, end = int(start), int(end)
    span = max(1, end - start)

    def fn(epoch: int) -> float:
        if epoch <= start:
            return float(v0)
        if epoch >= end:
            return float(v1)
        t = (epoch - start) / span
        return float(v0 + t * (v1 - v0))

    return fn


def cosine_ramp(start: int, end: int, v0: float, v1: float) -> Callable[[int], float]:
    """
    Cosine ramp: smooth start/end, monotone between v0->v1.
    """
    start, end = int(start), int(end)
    span = max(1, end - start)

    def fn(epoch: int) -> float:
        if epoch <= start:
            return float(v0)
        if epoch >= end:
            return float(v1)
        t = (epoch - start) / span
        t = 0.5 * (1 - math.cos(math.pi * t))
        return float(v0 + t * (v1 - v0))

    return fn


def mask_keep_schedule(
    total_epochs: int,
    keep_start: float = 0.25,
    keep_end: float = 0.10,
    ramp: str = "cosine",
    start_epoch: int = 0,
) -> Callable[[int], float]:
    """
    Returns a function(epoch)->keep_ratio for MAE-style masking (fraction of tokens kept).
    """
    if ramp == "linear":
        f = linear_ramp(start_epoch, total_epochs, keep_start, keep_end)
    else:
        f = cosine_ramp(start_epoch, total_epochs, keep_start, keep_end)
    return f


def grl_lambda_schedule(
    warmup_start: int = 30,
    warmup_end: int = 100,
    max_lambda: float = 0.05,
    ramp: str = "linear",
) -> Callable[[int], float]:
    """
    Returns a function(epoch)->lambda for GRL.
    """
    if ramp == "cosine":
        return cosine_ramp(warmup_start, warmup_end, 0.0, max_lambda)
    return linear_ramp(warmup_start, warmup_end, 0.0, max_lambda)
