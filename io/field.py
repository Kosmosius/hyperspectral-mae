from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ...core.types import CanonicalGrid, NDArrayF
from ...core.grid import load_grid_csv
from ...core.cr import continuum_removed
from ..manifests import FieldManifest, validate_manifest_file


@dataclass(frozen=True)
class FieldRecord:
    record_id: str
    f_lambda: NDArrayF
    instrument: Optional[str] = None
    site: Optional[str] = None
    cr_lambda: Optional[NDArrayF] = None


class FieldDataset:
    """
    Canonical field spectra loader.

    Expects NPZ at `spectra_npz` with:
      - 'f'     : [M, K] reflectance on CanonicalGrid
      - 'ids'   : [M] string IDs
      - optional: 'instrument' [M] strings
      - optional: 'site' [M] strings

    __getitem__ returns dict analogous to LabLibraryDataset but with field metadata.
    """
    def __init__(self, manifest_path: str | Path, return_cr: bool = True):
        self.manifest: FieldManifest = validate_manifest_file(manifest_path)
        base = Path(manifest_path).parent

        self.grid: CanonicalGrid = load_grid_csv(base / self.manifest.canonical_grid_csv)

        npz_path = base / self.manifest.spectra_npz
        if not npz_path.is_file():
            raise FileNotFoundError(f"Field spectra NPZ not found: {npz_path}")
        with np.load(npz_path, allow_pickle=True) as z:
            f = z["f"].astype(np.float64)
            ids = z["ids"].astype(str)
            instrument = z["instrument"].astype(str) if "instrument" in z else None
            site = z["site"].astype(str) if "site" in z else None

        if f.ndim != 2 or f.shape[1] != self.grid.K:
            raise ValueError("Field 'f' must be [M,K] and match CanonicalGrid length")
        if ids.shape[0] != f.shape[0]:
            raise ValueError("'ids' must match 'f' length")
        if instrument is not None and instrument.shape[0] != f.shape[0]:
            raise ValueError("'instrument' must match 'f' length")
        if site is not None and site.shape[0] != f.shape[0]:
            raise ValueError("'site' must match 'f' length")

        self._f = f
        self._ids = ids
        self._instrument = instrument
        self._site = site
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
        return {
            "record_id": self._ids[idx],
            "f_lambda": f,
            "cr_lambda": cr,
            "instrument": self._instrument[idx] if self._instrument is not None else None,
            "site": self._site[idx] if self._site is not None else None,
            "grid": self.grid,
        }
