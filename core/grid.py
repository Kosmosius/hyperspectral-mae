from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import numpy as np
import numpy.typing as npt

from .types import CanonicalGrid, NDArrayB, NDArrayF
from .hashing import sha256_ndarray


def create_grid(start_nm: float, stop_nm: float, step_nm: float, name: str | None = None) -> CanonicalGrid:
    """
    Create a uniform canonical wavelength grid in nm.
    """
    if step_nm <= 0 or stop_nm <= start_nm:
        raise ValueError("Invalid grid bounds or step")
    # ensure inclusive stop within floating tolerance
    n_steps = int(np.floor((stop_nm - start_nm) / step_nm + 1e-9)) + 1
    lambdas = np.linspace(start_nm, start_nm + step_nm * (n_steps - 1), n_steps, dtype=np.float64)
    grid_name = name or f"{int(start_nm)}-{int(stop_nm)}nm_{int(step_nm)}nm"
    return CanonicalGrid(lambdas_nm=lambdas, name=grid_name, sha256=sha256_ndarray(lambdas))


def save_grid_csv(grid: CanonicalGrid, path: str | Path) -> None:
    """
    Save grid wavelengths to a CSV with header including name and hash.
    """
    p = Path(path)
    header = f"name={grid.name},sha256={grid.sha256}"
    np.savetxt(p, grid.lambdas_nm, delimiter=",", fmt="%.8f", header=header)


def load_grid_csv(path: str | Path, expected_name: str | None = None) -> CanonicalGrid:
    """
    Load a canonical grid from CSV produced by save_grid_csv.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        first = f.readline()
    # Try to parse header metadata
    name = expected_name or "unknown"
    sha = None
    if first.startswith("#"):
        meta = first[1:].strip()
        for item in meta.split(","):
            if "=" in item:
                k, v = item.split("=", 1)
                if k.strip() == "name":
                    name = v.strip()
                elif k.strip() == "sha256":
                    sha = v.strip()
    lambdas = np.loadtxt(p, delimiter=",", dtype=np.float64, comments="#")
    comp_sha = sha256_ndarray(lambdas)
    if sha is not None and sha != comp_sha:
        raise ValueError(f"Grid hash mismatch: file={sha}, computed={comp_sha}")
    return CanonicalGrid(lambdas_nm=lambdas, name=name, sha256=comp_sha)


def build_water_vapor_mask(grid: CanonicalGrid,
                           intervals_nm: Sequence[Tuple[float, float]] | None = None) -> NDArrayB:
    """
    Build a boolean mask for water-vapor/low-SNR regions (True means masked/invalid).

    Default intervals cover typical atmospheric absorption windows in reflective domain.
    """
    if intervals_nm is None:
        intervals_nm = [
            (1340.0, 1460.0),  # strong WV
            (1790.0, 1960.0),  # strong WV
            (2450.0, 2500.0),  # SNR dropoff & edge
        ]
    lam = grid.lambdas_nm
    mask = np.zeros_like(lam, dtype=bool)
    for a, b in intervals_nm:
        mask |= (lam >= a) & (lam <= b)
    return mask
