from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple
import numpy as np
import numpy.typing as npt


NDArrayF = npt.NDArray[np.floating]
NDArrayB = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class CanonicalGrid:
    """Canonical wavelength grid (nm) with provenance hash."""
    lambdas_nm: NDArrayF          # shape [K], strictly increasing
    name: str                     # e.g., "400-2500nm_3nm"
    sha256: str                   # hash of lambdas_nm contents

    def __post_init__(self):
        if self.lambdas_nm.ndim != 1 or self.lambdas_nm.size < 3:
            raise ValueError("CanonicalGrid.lambdas_nm must be 1-D with length >= 3")
        if not np.all(np.diff(self.lambdas_nm) > 0):
            raise ValueError("CanonicalGrid.lambdas_nm must be strictly increasing")

    @property
    def K(self) -> int:
        return int(self.lambdas_nm.size)

    @property
    def step_nm(self) -> float:
        diffs = np.diff(self.lambdas_nm)
        return float(np.median(diffs))

    @property
    def bounds_nm(self) -> Tuple[float, float]:
        return float(self.lambdas_nm[0]), float(self.lambdas_nm[-1])


@dataclass(frozen=True)
class SRF:
    """Single-band SRF curve in physical wavelength coordinates (nm)."""
    wavelength_nm: NDArrayF   # [M], monotonically increasing (not required uniform)
    response: NDArrayF        # [M], non-negative (unnormalized OK)
    meta: Dict[str, float] = field(default_factory=dict)  # e.g., center_nm, fwhm_nm, band_id

    def __post_init__(self):
        if self.wavelength_nm.ndim != 1 or self.response.ndim != 1:
            raise ValueError("SRF arrays must be 1-D")
        if self.wavelength_nm.size != self.response.size:
            raise ValueError("SRF wavelength_nm and response must have same length")
        if not np.all(np.diff(self.wavelength_nm) >= 0):
            raise ValueError("SRF.wavelength_nm must be non-decreasing")
        if np.any(self.response < 0):
            raise ValueError("SRF.response must be non-negative")

    @property
    def center_nm(self) -> Optional[float]:
        return self.meta.get("center_nm")

    @property
    def fwhm_nm(self) -> Optional[float]:
        return self.meta.get("fwhm_nm")


@dataclass(frozen=True)
class SRFMatrix:
    """SRF mixing matrix R (row-stochastic), mapping canonical spectra f[K] to bands y[N]."""
    R: NDArrayF                          # [N, K], rows sum to 1 within tolerance
    centers_nm: Optional[NDArrayF] = None
    fwhm_nm: Optional[NDArrayF] = None
    effective_width_bins_: Optional[NDArrayF] = None  # 1/∑ r_i^2
    grid_name: Optional[str] = None
    sha256: Optional[str] = None
    meta: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        if self.R.ndim != 2 or self.R.shape[0] < 1 or self.R.shape[1] < 3:
            raise ValueError("SRFMatrix.R must be [N,K] with N>=1, K>=3")
        if np.any(self.R < 0):
            raise ValueError("SRFMatrix.R must be non-negative")


@dataclass(frozen=True)
class Sample:
    """A single observed spectrum in band-space with its SRF operator."""
    y: NDArrayF                 # [N] band-averaged values (reflectance preferred)
    R: SRFMatrix                # operator (row-stochastic)
    mask: Optional[NDArrayB] = None  # valid-band mask; if None, all bands valid
    units: str = "reflectance"       # "reflectance" | "radiance"

    def __post_init__(self):
        if self.y.ndim != 1 or self.y.shape[0] != self.R.R.shape[0]:
            raise ValueError("Sample.y length must match number of rows in R")
        if self.mask is not None and (self.mask.ndim != 1 or self.mask.shape[0] != self.y.shape[0]):
            raise ValueError("Sample.mask must be 1-D of same length as y")


@dataclass(frozen=True)
class SplineBasis:
    """B-spline basis sampled on canonical grid."""
    B: NDArrayF                 # [K, M], columns are basis functions at grid points (optionally orthonormal)
    knots_nm: NDArrayF          # knot vector in nm (internal representation)
    degree: int                 # spline degree (e.g., 3)
    orthonormal: bool           # True if columns are orthonormal under trapezoidal inner product
    cond_number: float          # numeric stability indicator
    grid_name: str              # link to CanonicalGrid.name
    sha256: str                 # hash of contents + metadata


@dataclass(frozen=True)
class SRFProjectionBasis:
    """Basis to project SRF shapes (columns orthonormal under trapezoidal rule)."""
    S: NDArrayF                 # [K, Ms]
    degree: int
    knots_nm: NDArrayF
    grid_name: str
    sha256: str
