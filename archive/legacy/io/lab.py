from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from ...core.types import CanonicalGrid, NDArrayF
from ...core.grid import load_grid_csv
from ...core.cr import continuum_removed
from ..manifests import LabManifest, validate_manifest_file


@dataclass(frozen=True)
class LabRecord:
    """A single canonical lab spectrum and metadata."""
    spectrum_id: str
    f_lambda: NDArrayF    # [K] reflectance on CanonicalGrid
    family: Optional[str] = None
    cr_lambda: Optional[NDArrayF] = None     # optionally precomputed CR for convenience


class LabLibraryDataset:
    """
    Canonical lab library dataset (USGS/ECOSTRESS/ASTER) loader.

    Expects an NPZ at `spectra_npz` with keys:
      - 'f'   : float32/64 array [M, K] canonicalized reflectance
      - 'ids' : array of strings [M]
      - optional: 'family' [M] strings

    The manifest must also provide a canonical grid CSV.

    __getitem__ returns:
      dict(
        spectrum_id: str,
        f_lambda: np.ndarray[K],
        cr_lambda: np.ndarray[K] or None,
        family: str or None,
        grid: CanonicalGrid (shared)
      )
    """
    def __init__(
        self,
        manifest_path: str | Path,
        return_cr: bool = True,
    ):
        self.manifest: LabManifest = validate_manifest_file(manifest_path)  # raises on error
        base = Path(manifest_path).parent

        # Load grid
        self.grid: CanonicalGrid = load_grid_csv(base / self.manifest.canonical_grid_csv)

        # Load spectra
        npz_path = base / self.manifest.spectra_npz
        if not npz_path.is_file():
            raise FileNotFoundError(f"Lab spectra NPZ not found: {npz_path}")
        with np.load(npz_path, allow_pickle=True) as z:
            f = z["f"].astype(np.float64)  # [M,K]
            ids = z["ids"]
            families = z["family"] if "family" in z else None

        if f.ndim != 2 or f.shape[1] != self.grid.K:
            raise ValueError("Spectra 'f' must be [M, K] and match CanonicalGrid length")
        if ids.shape[0] != f.shape[0]:
            raise ValueError("'ids' length must match 'f' rows")
        if families is not None and families.shape[0] != f.shape[0]:
            raise ValueError("'family' length must match 'f' rows")

        self._f = f
        self._ids = ids.astype(str)
        self._families = families.astype(str) if families is not None else None
        self._return_cr = return_cr
        self._cr_cache: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return int(self._f.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, object]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        f = self._f[idx]
        cr = None
        if self._return_cr:
            if idx not in self._cr_cache:
                self._cr_cache[idx] = continuum_removed(self.grid.lambdas_nm, f)
            cr = self._cr_cache[idx]
        out = {
            "spectrum_id": self._ids[idx],
            "f_lambda": f,
            "cr_lambda": cr,
            "family": self._families[idx] if self._families is not None else None,
            "grid": self.grid,
        }
        return out

    @property
    def K(self) -> int:
        return self.grid.K

    @property
    def families(self) -> Optional[List[str]]:
        if self._families is None:
            return None
        return list(sorted(set(self._families.tolist())))
