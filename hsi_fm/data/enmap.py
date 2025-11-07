"""EnMAP product utilities."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import math

import numpy as np
import torch
from torch.nn import functional as F


Tensor = torch.Tensor


DEFAULT_ENMAP_CENTERS_VNIR = torch.linspace(420.0, 1000.0, steps=5)
DEFAULT_ENMAP_CENTERS_SWIR = torch.linspace(1000.0, 2450.0, steps=5)
DEFAULT_ENMAP_FWHM_VNIR = torch.full((5,), 10.0)
DEFAULT_ENMAP_FWHM_SWIR = torch.full((5,), 12.0)


@dataclass
class EnMAPSample:
    cube: Tensor
    wavelengths_nm: Tensor
    geometry: Dict[str, Tensor]
    qa_mask: torch.BoolTensor
    sensor_id: str = "EnMAP"
    srf: Optional[Tensor] = None


DatasetLoader = Callable[[Path], Any]


def _default_loader(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        data = np.load(path)

        class _Wrapper:
            def __init__(self, data: np.lib.npyio.NpzFile):
                self._data = data

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                self._data.close()
                return False

            @property
            def variables(self):
                return dict(self._data)

        return _Wrapper(data)
    try:  # pragma: no cover - optional dependency
        import h5py  # type: ignore

        return h5py.File(path, "r")
    except Exception as err:  # pragma: no cover - optional path
        raise RuntimeError("Install h5py/netCDF4 for EnMAP products") from err


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "__array__"):
        return np.asarray(value)
    if hasattr(value, "value"):
        return np.asarray(value.value)
    if hasattr(value, "__getitem__"):
        return np.asarray(value[...])
    raise TypeError(f"Unsupported array type: {type(value)!r}")


def _stack_defaults() -> Tuple[Tensor, Tensor]:
    centers = torch.cat([DEFAULT_ENMAP_CENTERS_VNIR, DEFAULT_ENMAP_CENTERS_SWIR])
    fwhm = torch.cat([DEFAULT_ENMAP_FWHM_VNIR, DEFAULT_ENMAP_FWHM_SWIR])
    return centers, fwhm


def _get_variable(dataset: Any, key: str) -> Any:
    variables = getattr(dataset, "variables", {})
    if isinstance(variables, Mapping) or hasattr(variables, "__contains__"):
        if key in variables:
            return variables[key]
    if hasattr(dataset, key):
        return getattr(dataset, key)
    return None


def _extract_geometry(source: Any) -> Dict[str, Tensor]:
    fields = {
        "sza": ["solar_zenith", "SunZenith", "Sun_Zenith"],
        "vza": ["sensor_zenith", "ViewZenith", "Sensor_Zenith"],
        "raa": ["relative_azimuth", "RelativeAzimuth"],
        "smile": ["smile", "Smile"]
    }
    geometry: Dict[str, Tensor] = {}
    for key, aliases in fields.items():
        value = None
        for alias in aliases:
            value = _get_variable(source, alias)
            if value is not None:
                break
        if value is not None:
            geometry[key] = torch.as_tensor(_to_numpy(value)).float().reshape(-1)[0]
    return geometry


def _extract_wavelengths(dataset: Any, candidates: Iterable[str]) -> Tensor:
    for name in candidates:
        value = _get_variable(dataset, name)
        if value is not None:
            return torch.as_tensor(_to_numpy(value)).float()
    centers, _ = _stack_defaults()
    return centers.clone()


def _extract_cube(dataset: Any) -> Tensor:
    for key in ("reflectance", "cube", "radiance"):
        value = _get_variable(dataset, key)
        if value is not None:
            cube = torch.as_tensor(_to_numpy(value)).float()
            break
    else:
        raise KeyError("No cube variable found in EnMAP product")
    if cube.ndim == 2:
        cube = cube.unsqueeze(0)
    if cube.ndim != 3:
        raise ValueError("Cube must be (bands, height, width)")
    return cube


def _extract_mask(dataset: Any) -> torch.BoolTensor:
    for name in ("mask", "qa_mask", "quality"):
        value = _get_variable(dataset, name)
        if value is not None:
            return torch.as_tensor(_to_numpy(value)).bool()
    return torch.ones(1, dtype=torch.bool)


def _gaussian_from_table(centers: Tensor, fwhm: Tensor, wavelengths: Tensor) -> Tensor:
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    diff = wavelengths.unsqueeze(0) - centers.unsqueeze(-1)
    weights = torch.exp(-0.5 * (diff / sigma.unsqueeze(-1)) ** 2)
    return F.normalize(weights, p=1, dim=-1)


def load_enmap_srf_table(path: Path) -> Tuple[Tensor, Tensor]:
    table = np.loadtxt(path, delimiter=",", skiprows=1)
    centers = torch.as_tensor(table[:, 0]).float()
    fwhm = torch.as_tensor(table[:, 1]).float()
    return centers, fwhm


def export_srf(centers: Tensor, fwhm: Tensor, destination: Path) -> None:
    data = torch.stack([centers, fwhm], dim=-1).cpu().numpy()
    header = "center_nm,fwhm_nm"
    np.savetxt(destination, data, delimiter=",", header=header, comments="")


def read_enmap_l2a(
    path: Path | str,
    *,
    loader: Optional[DatasetLoader] = None,
    srf_table: Optional[Tuple[Tensor, Tensor]] = None,
) -> EnMAPSample:
    loader = loader or _default_loader
    fallback = srf_table or _stack_defaults()
    with loader(Path(path)) as dataset:
        cube = _extract_cube(dataset)
        wavelengths = _extract_wavelengths(dataset, ["wavelengths", "band_centers"])
        qa_mask = _extract_mask(dataset)
        geometry = _extract_geometry(dataset)
        centers, fwhm = fallback
        srf = _gaussian_from_table(centers, fwhm, wavelengths)
    return EnMAPSample(cube=cube, wavelengths_nm=wavelengths, geometry=geometry, qa_mask=qa_mask, srf=srf)


__all__ = [
    "EnMAPSample",
    "read_enmap_l2a",
    "load_enmap_srf_table",
    "export_srf",
]

