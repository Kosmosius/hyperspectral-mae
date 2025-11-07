"""Mock LWIR teacher implementation."""
from __future__ import annotations

from typing import Dict

import torch


def estimate_lwir_teacher(cube: torch.Tensor) -> Dict[str, torch.Tensor]:
    baseline = cube.mean(dim=(1, 2), keepdim=True)
    residual = cube - baseline
    score = residual.std(dim=0)
    return {"cmf": score, "glrt": score}


__all__ = ["estimate_lwir_teacher"]
