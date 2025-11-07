from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np

from ...core.types import CanonicalGrid, SRFMatrix, Sample, NDArrayB, NDArrayF
from ...core.grid import load_grid_csv, build_water_vapor_mask
from ...core.renderer import load_R_npz
from ..manifests import SatelliteManifest, validate_manifest_file


@dataclass(frozen=True)
class PixelRef:
    """Reference to a pixel in a cube."""
    row: int
    col: int
    scene_id: str


class SatellitePixelDataset:
    """
    Satellite pixel sampler that returns per-pixel band vectors y[N], SRFMatrix R[N,K], and band masks.

    Expected cube NPZ structure (minimal, mission-agnostic):
      - 'y'           : float32/64 array [H, W, N] band-averaged values (reflectance or radiance)
      - optional 'valid' : bool array [H, W] (true=valid pixel)
      - optional 'mask_y': bool array [H, W, N] (true=invalid band for this pixel)
      - optional 'units' : str (e.g., "reflectance" or "radiance")
      - optional 'scene_id' : str
      - optional 'meta': pickled dict with ancillary info

    SRF matrix NPZ must be mission-level SRF on the CanonicalGrid (see core.renderer.save_R_npz).
    """
    def __init__(
        self,
        manifest_path: str | Path,
        sample_mode: str = "all",   # "all" | "random" | "grid"
        max_samples: Optional[int] = None,
        random_seed: int = 17,
        include_wv_mask: bool = True,
    ):
        self.manifest: SatelliteManifest = validate_manifest_file(manifest_path)
        base = Path(manifest_path).parent

        # Load grid and SRF matrix (mission-level)
        self.grid: CanonicalGrid = load_grid_csv(base / self.manifest.canonical_grid_csv)
        self.R: SRFMatrix = load_R_npz(base / self.manifest.srf_matrix_npz)

        # Load cube
        cube = np.load(base / self.manifest.cube_npz, allow_pickle=True)
        self._y = cube["y"].astype(np.float64)           # [H,W,N]
        self._H, self._W, self._N = self._y.shape

        self._valid = cube["valid"].astype(bool) if "valid" in cube else np.ones((self._H, self._W), dtype=bool)
        self._mask_y = cube["mask_y"].astype(bool) if "mask_y" in cube else None
        self._units = str(cube["units"]) if "units" in cube else "reflectance"
        self._scene_id = str(cube["scene_id"]) if "scene_id" in cube else "scene"

        # Build default WV mask on canonical grid; will apply AFTER rendering/decode phases in metrics,
        # but for band-space losses, we honor per-band validity first, then optional WV mask mapped by mission bands.
        self._wv_mask_canonical = build_water_vapor_mask(self.grid) if include_wv_mask else None

        # Validate band dimension with SRF matrix rows
        if self.R.R.shape[0] != self._N:
            raise ValueError(f"Band count mismatch: cube N={self._N} vs SRF rows={self.R.R.shape[0]}")

        # Prepare list of candidate pixels
        self._pixels: List[PixelRef] = []
        if sample_mode == "all":
            rows, cols = np.where(self._valid)
            for r, c in zip(rows.tolist(), cols.tolist()):
                self._pixels.append(PixelRef(r, c, self._scene_id))
        elif sample_mode == "grid":
            step_r = max(1, int(np.sqrt((self._H * self._W) / 4096)))  # ~64x64 samples default
            step_c = step_r
            for r in range(0, self._H, step_r):
                for c in range(0, self._W, step_c):
                    if self._valid[r, c]:
                        self._pixels.append(PixelRef(r, c, self._scene_id))
        elif sample_mode == "random":
            rng = np.random.default_rng(random_seed)
            rows, cols = np.where(self._valid)
            idx = np.arange(rows.shape[0])
            rng.shuffle(idx)
            for i in idx.tolist():
                self._pixels.append(PixelRef(int(rows[i]), int(cols[i]), self._scene_id))
        else:
            raise ValueError(f"Unknown sample_mode: {sample_mode}")

        if max_samples is not None:
            self._pixels = self._pixels[: int(max_samples)]

    def __len__(self) -> int:
        return len(self._pixels)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        ref = self._pixels[idx]
        y = self._y[ref.row, ref.col, :].copy()  # [N]

        # Per-pixel band validity
        if self._mask_y is not None:
            band_valid = ~self._mask_y[ref.row, ref.col, :].astype(bool)
        else:
            band_valid = np.ones(self._N, dtype=bool)

        # Optional (mission-agnostic) WV masking on canonical grid: we cannot apply directly here since
        # y is band-space; downstream training uses SRF and canonical masks consistently.
        # We include the per-band validity now; canonical WV mask is made available in 'extra'.
        sample = {
            "pixel": {"row": ref.row, "col": ref.col, "scene_id": ref.scene_id},
            "y": y,
            "R": self.R,               # SRFMatrix on canonical grid
            "mask": band_valid,        # valid bands (True = valid)
            "units": self._units,
            "grid": self.grid,
            "extra": {
                "wv_mask_canonical": self._wv_mask_canonical,
                "mission": self.manifest.mission or "unknown",
            },
        }
        return sample


# Short aliases for mission-specific configs (if you want to customize per mission later)
PRISMA = SatellitePixelDataset
EnMAP = SatellitePixelDataset
