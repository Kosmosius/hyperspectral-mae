"""Mock SWIR teacher implementation."""
from __future__ import annotations

from typing import Dict

import torch


def estimate_swir_teacher(cube: torch.Tensor) -> Dict[str, torch.Tensor]:
    mean = cube.mean(dim=0, keepdim=True)
    anomaly = cube - mean
    score = torch.clamp(anomaly.mean(dim=0), min=0.0)
    return {"ctmf": score, "imap": score}


__all__ = ["estimate_swir_teacher"]
