from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

try:
    # Prefer canonical operators when available
    from ..core.renderer import build_R as core_build_R
except Exception:
    core_build_R = None  # fallback to discrete normalization

ShapeName = Literal["box", "tri", "gauss", "skewed"]


@dataclass
class EmulationConfig:
    bands_min: int = 20
    bands_max: int = 150
    width_nm_min: float = 5.0
    width_nm_max: float = 80.0
    jitter_nm: float = 2.0
    tail_leakage: float = 0.01     # fraction of uniform leakage inside [λmin, λmax]
    p_shapes: Tuple[float, float, float, float] = (0.25, 0.25, 0.4, 0.10)  # box,tri,gauss,skewed
    dropout_srf_shape_prob: float = 0.15  # sometimes omit shape tokens to test robustness


@dataclass
class EmulationOutput:
    centers_nm: np.ndarray       # [N]
    widths_nm: np.ndarray        # [N]
    shapes: List[ShapeName]      # [N]
    srfs: List[Dict[str, np.ndarray]]  # list of {'wavelength': [K], 'response': [K]} discrete curves on canonical λ
    R: np.ndarray                # [N,K] normalized weights for rendering
    meta: Dict[str, np.ndarray]  # auxiliary (e.g., effective width, tails, jitter used)


class RandomSensorEmulator:
    """
    Generates synthetic SRFs on top of a canonical wavelength grid.
    Produces discrete response curves and an [N,K] renderer matrix R
    (normalized weights) suitable for y = R @ f.
    """
    def __init__(self, canon_wavelength_nm: np.ndarray, cfg: EmulationConfig = EmulationConfig()):
        self.w = canon_wavelength_nm.astype(np.float64)  # [K]
        assert np.all(np.diff(self.w) > 0), "Canonical grid must be strictly increasing"
        self.cfg = cfg

    # ---------- SRF primitives ----------

    def _srf_box(self, c: float, fwhm: float) -> np.ndarray:
        half = 0.5 * fwhm
        r = ((self.w >= c - half) & (self.w <= c + half)).astype(np.float64)
        return r

    def _srf_tri(self, c: float, fwhm: float) -> np.ndarray:
        half = 0.5 * fwhm
        dist = np.abs(self.w - c)
        r = np.clip(1.0 - dist / half, 0.0, 1.0)
        return r

    def _srf_gauss(self, c: float, fwhm: float) -> np.ndarray:
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        r = np.exp(-0.5 * ((self.w - c) / sigma) ** 2)
        return r

    def _srf_skewed(self, c: float, fwhm: float, skew: float = 0.3) -> np.ndarray:
        """
        Skewed Gaussian via different left/right sigmas.
        """
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        left_sigma = sigma * (1.0 + skew)
        right_sigma = sigma * (1.0 - skew)
        r = np.zeros_like(self.w, dtype=np.float64)
        left = self.w <= c
        right = ~left
        r[left] = np.exp(-0.5 * ((self.w[left] - c) / left_sigma) ** 2)
        r[right] = np.exp(-0.5 * ((self.w[right] - c) / right_sigma) ** 2)
        return r

    def _sample_centers(self, N: int) -> np.ndarray:
        # Uniform across the canonical support with jitter
        lam_min, lam_max = float(self.w[0]), float(self.w[-1])
        centers = np.random.uniform(lam_min, lam_max, size=N)
        if self.cfg.jitter_nm > 0:
            centers = centers + np.random.uniform(-self.cfg.jitter_nm, self.cfg.jitter_nm, size=N)
        return centers

    def _sample_widths(self, N: int) -> np.ndarray:
        widths = np.random.uniform(self.cfg.width_nm_min, self.cfg.width_nm_max, size=N)
        return widths

    def _sample_shapes(self, N: int) -> List[ShapeName]:
        probs = np.array(self.cfg.p_shapes, dtype=np.float64)
        probs = probs / probs.sum()
        names: List[ShapeName] = ["box", "tri", "gauss", "skewed"]
        idx = np.random.choice(len(names), size=N, p=probs)
        return [names[i] for i in idx]

    # ---------- Public API ----------

    def sample(self, N: Optional[int] = None) -> EmulationOutput:
        if N is None:
            N = int(np.random.randint(self.cfg.bands_min, self.cfg.bands_max + 1))
        centers = self._sample_centers(N)
        widths = self._sample_widths(N)
        shapes = self._sample_shapes(N)

        curves: List[Dict[str, np.ndarray]] = []
        weights = []
        for c, w, sh in zip(centers, widths, shapes):
            if sh == "box":
                r = self._srf_box(c, w)
            elif sh == "tri":
                r = self._srf_tri(c, w)
            elif sh == "gauss":
                r = self._srf_gauss(c, w)
            else:
                r = self._srf_skewed(c, w, skew=0.3)

            # optional low-energy uniform leakage inside support
            if self.cfg.tail_leakage > 0:
                leakage = self.cfg.tail_leakage * np.ones_like(r) / len(r)
                r = r + leakage

            # normalize per-band to sum to 1
            s = r.sum()
            r = r / s if s > 0 else r

            curves.append({"wavelength": self.w, "response": r})
            weights.append(r)

        R = np.stack(weights, axis=0)  # [N,K]
        eff_width = 1.0 / np.maximum(1e-12, (R ** 2).sum(axis=1))  # top-hat equivalent

        return EmulationOutput(
            centers_nm=centers.astype(np.float64),
            widths_nm=widths.astype(np.float64),
            shapes=shapes,
            srfs=curves,
            R=R,
            meta={"effective_width_nm": eff_width.astype(np.float64)},
        )


class CatalogBiasedEmulator(RandomSensorEmulator):
    """
    Optional: bias sampling from a catalog of SRF centers/widths distributions while retaining
    randomized shapes. If a catalog is not provided for a draw, it falls back to RandomSensorEmulator.
    """
    def __init__(
        self,
        canon_wavelength_nm: np.ndarray,
        cfg: EmulationConfig = EmulationConfig(),
        catalog_centers_nm: Optional[Sequence[np.ndarray]] = None,
        catalog_widths_nm: Optional[Sequence[np.ndarray]] = None,
    ):
        super().__init__(canon_wavelength_nm, cfg)
        self._cat_centers = list(catalog_centers_nm or [])
        self._cat_widths = list(catalog_widths_nm or [])

    def _sample_centers(self, N: int) -> np.ndarray:
        if self._cat_centers:
            arr = self._cat_centers[np.random.randint(0, len(self._cat_centers))]
            if arr.size >= N:
                idx = np.random.choice(arr.size, size=N, replace=False)
                out = arr[idx]
            else:
                out = np.random.choice(arr, size=N, replace=True)
            if self.cfg.jitter_nm > 0:
                out = out + np.random.uniform(-self.cfg.jitter_nm, self.cfg.jitter_nm, size=N)
            return out
        return super()._sample_centers(N)

    def _sample_widths(self, N: int) -> np.ndarray:
        if self._cat_widths:
            arr = self._cat_widths[np.random.randint(0, len(self._cat_widths))]
            if arr.size >= N:
                idx = np.random.choice(arr.size, size=N, replace=False)
                out = arr[idx]
            else:
                out = np.random.choice(arr, size=N, replace=True)
            return out
        return super()._sample_widths(N)
