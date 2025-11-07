"""Utilities for sampling tube patches from hyperspectral cubes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import torch


@dataclass
class PatchConfig:
    """Configuration for extracting patches from hyperspectral cubes."""

    height: int
    width: int
    stride: int


class PatchSampler:
    """Sample spatial tube patches from a cube.

    Parameters
    ----------
    patch_config:
        Dimensions of the extracted patches.
    random_offset:
        If ``True`` a random offset within the stride is used to introduce
        stochasticity during training.
    """

    def __init__(self, patch_config: PatchConfig, random_offset: bool = True):
        self.config = patch_config
        self.random_offset = random_offset

    def sample(self, cube: torch.Tensor) -> Iterator[torch.Tensor]:
        """Yield tube patches from ``cube``.

        The cube is expected to be shaped as ``(C, H, W)``.
        """

        if cube.ndim != 3:
            raise ValueError("cube must be 3-dimensional (C, H, W)")
        c, h, w = cube.shape
        ph, pw, stride = self.config.height, self.config.width, self.config.stride

        if self.random_offset:
            offset_h = torch.randint(0, stride, ()).item()
            offset_w = torch.randint(0, stride, ()).item()
        else:
            offset_h = 0
            offset_w = 0

        for top in range(offset_h, max(h - ph + 1, 1), stride):
            for left in range(offset_w, max(w - pw + 1, 1), stride):
                patch = cube[:, top : top + ph, left : left + pw]
                if patch.shape[1] == ph and patch.shape[2] == pw:
                    yield patch


__all__ = ["PatchConfig", "PatchSampler"]
