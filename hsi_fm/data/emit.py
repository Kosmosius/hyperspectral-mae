"""EMIT product readers with minimal dependencies."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import math

import numpy as np
import torch
from collections.abc import Mapping

from torch.nn import functional as F


Tensor = torch.Tensor


DEFAULT_EMIT_CENTERS_NM = torch.linspace(400.0, 2500.0, steps=10)
DEFAULT_EMIT_FWHM_NM = torch.full((10,), 10.0)


@dataclass
class EMITSample:
    cube: Tensor
    wavelengths_nm: Tensor
    geometry: Dict[str, Tensor]
    qa_mask: torch.BoolTensor
    sensor_id: str = "EMIT"
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
                return self._data

            def get(self, key: str, default=None):
                return self._data[key] if key in self._data else default

        return _Wrapper(data)
    try:
        import h5py  # type: ignore

        return h5py.File(path, "r")
    except Exception as err:  # pragma: no cover - best effort
        raise RuntimeError(
            "Unable to open EMIT product. Install netCDF4 or h5py, or provide a custom loader."
        ) from err


def _to_numpy(array: Any) -> np.ndarray:
    if array is None:
        raise ValueError("Missing array in EMIT product")
    if isinstance(array, np.ndarray):
        return array
    if hasattr(array, "__array__"):
        return np.asarray(array)
    if hasattr(array, "value"):
        return np.asarray(array.value)
    if hasattr(array, "__getitem__"):
        return np.asarray(array[...])
    raise TypeError(f"Unsupported array type {type(array)!r}")


def _extract_geometry(source: Any) -> Dict[str, Tensor]:
    keys = {
        "sza": ["solar_zenith", "SolarZenith", "solar_zenith_angle"],
        "vza": ["view_zenith", "ViewZenith", "sensor_zenith"],
        "raa": ["relative_azimuth", "RelativeAzimuth"],
        "aot": ["aerosol_optical_depth", "AOT"],
        "alt": ["altitude", "surface_altitude"],
    }
    geometry: Dict[str, Tensor] = {}
    for key, aliases in keys.items():
        value = None
        for alias in aliases:
            if hasattr(source, alias):
                value = getattr(source, alias)
                break
            if isinstance(getattr(source, "variables", {}), dict) and alias in source.variables:
                value = source.variables[alias]
                break
        if value is not None:
            arr = torch.as_tensor(np.asarray(value)).float()
            geometry[key] = arr.reshape(-1)[0]
    return geometry


def _get_variable(dataset: Any, key: str) -> Any:
    variables = getattr(dataset, "variables", {})
    if isinstance(variables, Mapping) or hasattr(variables, "__contains__"):
        if key in variables:
            return variables[key]
    if hasattr(dataset, key):
        return getattr(dataset, key)
    return None


def _extract_wavelengths(dataset: Any, candidates: Iterable[str]) -> Tensor:
    for name in candidates:
        value = _get_variable(dataset, name)
        if value is not None:
            return torch.as_tensor(_to_numpy(value)).float()
    return DEFAULT_EMIT_CENTERS_NM.clone()


def _extract_cube(dataset: Any, key: str) -> Tensor:
    array = _get_variable(dataset, key)
    if array is None:
        raise KeyError(f"Cube variable '{key}' not found in EMIT product")
    cube = torch.as_tensor(_to_numpy(array)).float()
    if cube.ndim == 2:
        cube = cube.unsqueeze(0)
    if cube.ndim != 3:
        raise ValueError("Cube must have shape (bands, height, width)")
    return cube


def _extract_mask(dataset: Any) -> torch.BoolTensor:
    for name in ("qa_mask", "quality_mask", "mask"):
        value = _get_variable(dataset, name)
        if value is not None:
            return torch.as_tensor(_to_numpy(value)).bool()
    return torch.ones(1, dtype=torch.bool)


def load_emit_srf_table(path: Path) -> Tuple[Tensor, Tensor]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    centers = torch.as_tensor(data[:, 0]).float()
    fwhm = torch.as_tensor(data[:, 1]).float()
    return centers, fwhm


def _extract_srf(dataset: Any, fallback: Tuple[Tensor, Tensor]) -> Tensor:
    variables = getattr(dataset, "variables", {})
    if (isinstance(variables, Mapping) or hasattr(variables, "__contains__")) and "srf" in variables:
        return torch.as_tensor(_to_numpy(variables["srf"])).float()
    centers, fwhm = fallback
    wavelengths = _extract_wavelengths(dataset, ["wavelengths", "band_centers"])
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    diff = wavelengths.unsqueeze(0) - centers.unsqueeze(-1)
    weights = torch.exp(-0.5 * (diff / sigma.unsqueeze(-1)) ** 2)
    return F.normalize(weights, p=1, dim=-1)


def read_emit_product(
    path: Path,
    *,
    data_key: str,
    loader: Optional[DatasetLoader] = None,
    srf_table: Optional[Tuple[Tensor, Tensor]] = None,
) -> EMITSample:
    loader = loader or _default_loader
    fallback = srf_table or (DEFAULT_EMIT_CENTERS_NM, DEFAULT_EMIT_FWHM_NM)
    with loader(Path(path)) as dataset:
        cube = _extract_cube(dataset, data_key)
        wavelengths = _extract_wavelengths(dataset, ["wavelengths", "band_centers"])
        qa_mask = _extract_mask(dataset)
        geometry = _extract_geometry(dataset)
        srf = _extract_srf(dataset, fallback)
    return EMITSample(cube=cube, wavelengths_nm=wavelengths, geometry=geometry, qa_mask=qa_mask, srf=srf)


def read_emit_l2a(path: Path | str, **kwargs: Any) -> EMITSample:
    return read_emit_product(Path(path), data_key="reflectance", **kwargs)


def read_emit_l1b(path: Path | str, **kwargs: Any) -> EMITSample:
    return read_emit_product(Path(path), data_key="radiance", **kwargs)


__all__ = [
    "EMITSample",
    "read_emit_l1b",
    "read_emit_l2a",
    "read_emit_product",
    "load_emit_srf_table",
]

