"""Radiative transfer adapters (placeholders for external engines)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Sequence

import torch


@dataclass
class Py6SConfig:
    aerosol: str = "continental"
    water_vapor: float = 2.0
    ozone: float = 0.3


def run_py6s(config: Py6SConfig, wavelengths: torch.Tensor, reflectance: torch.Tensor) -> torch.Tensor:
    """Placeholder Py6S adapter.

    In production this function would wrap the Py6S executable. For the scaffold we
    simply return the input reflectance scaled by a nominal factor, acting as an
    energy balance placeholder.
    """

    scale = 0.8 if config.aerosol == "continental" else 0.75
    return reflectance * scale


__all__ = ["Py6SConfig", "run_py6s"]
