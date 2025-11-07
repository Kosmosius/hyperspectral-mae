"""Dataset abstractions for hyperspectral cubes and lab spectra."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class SceneMetadata:
    """Metadata describing a hyperspectral scene.

    Attributes
    ----------
    sensor: str
        Sensor name (e.g. ``"EMIT"``).
    wavelengths: Tensor
        Wavelength centers in microns.
    sza: float
        Solar zenith angle in degrees.
    vza: float
        View zenith angle in degrees.
    raa: float
        Relative azimuth angle in degrees.
    extra: Mapping[str, float]
        Additional parameters (e.g. aerosol optical thickness).
    """

    sensor: str
    wavelengths: np.ndarray
    sza: float
    vza: float
    raa: float
    extra: Mapping[str, float]


class HyperspectralCube(Dataset[Mapping[str, torch.Tensor]]):
    """Generic dataset wrapper around hyperspectral cubes.

    The class abstracts away the sensor-specific loading logic. Implementations
    are expected to subclass and override :meth:`_load_cube` to return a tensor
    shaped as ``(C, H, W)`` in radiance or reflectance units.
    """

    def __init__(self, paths: Iterable[Path], metadata: Iterable[SceneMetadata]):
        self._paths = list(paths)
        self._metadata = list(metadata)
        if len(self._paths) != len(self._metadata):
            raise ValueError("paths and metadata must have the same length")

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._paths)

    def __getitem__(self, idx: int) -> Mapping[str, torch.Tensor]:
        cube = self._load_cube(self._paths[idx])
        info = self._metadata[idx]
        wavelengths = torch.as_tensor(info.wavelengths, dtype=cube.dtype, device=cube.device)
        angles = torch.tensor([info.sza, info.vza, info.raa], dtype=torch.float32, device=cube.device)
        extra_values = (
            torch.tensor(list(info.extra.values()), dtype=torch.float32, device=cube.device)
            if info.extra
            else torch.zeros(0, dtype=torch.float32, device=cube.device)
        )
        return {
            "cube": cube,
            "metadata": {
                "sensor": info.sensor,
                "wavelengths": wavelengths,
                "angles": angles,
                "extra": extra_values,
            },
        }

    def _load_cube(self, path: Path) -> torch.Tensor:
        raise NotImplementedError


class LabSpectraDataset(Dataset[Mapping[str, torch.Tensor]]):
    """Dataset for lab-based reflectance spectra."""

    def __init__(self, spectra: Tensor, labels: Optional[List[str]] = None):
        if spectra.ndim != 2:
            raise ValueError("spectra must be 2D")
        self._spectra = spectra.to(dtype=torch.float32)
        self._labels = labels or [""] * len(spectra)
        if len(self._labels) != len(spectra):
            raise ValueError("labels length mismatch")

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._spectra)

    def __getitem__(self, idx: int) -> Mapping[str, torch.Tensor]:
        return {
            "spectrum": self._spectra[idx],
            "label": self._labels[idx],
        }


class CachedIterableDataset(Dataset[Mapping[str, torch.Tensor]]):
    """Materialized view over an iterator dataset.

    This helper converts an arbitrary iterator of samples into a dataset with
    deterministic random access, which is useful for caching expensive physics
    augmentations.
    """

    def __init__(self, iterator: Iterator[Mapping[str, torch.Tensor]]):
        self._items: List[Mapping[str, torch.Tensor]] = list(iterator)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._items)

    def __getitem__(self, idx: int) -> Mapping[str, torch.Tensor]:
        return self._items[idx]


__all__ = [
    "SceneMetadata",
    "HyperspectralCube",
    "LabSpectraDataset",
    "CachedIterableDataset",
]
