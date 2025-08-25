from __future__ import annotations

from typing import Iterable, List, Optional, Tuple
import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d

from .types import CanonicalGrid, SRF, SRFMatrix, SRFProjectionBasis, NDArrayF
from .hashing import sha256_ndarray, stable_meta_hash


def gaussian_srf(lambdas_nm: NDArrayF, center_nm: float, fwhm_nm: float) -> NDArrayF:
    """
    Unnormalized Gaussian SRF with given center and FWHM.
    """
    sigma = float(fwhm_nm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return np.exp(-0.5 * ((lambdas_nm - center_nm) / sigma) ** 2)


def triangular_srf(lambdas_nm: NDArrayF, center_nm: float, fwhm_nm: float) -> NDArrayF:
    """
    Symmetric triangular SRF. We define base width = 2 * fwhm so that FWHM matches spec.
    Peak is 1 at center; linearly decays to 0 at center ± base_width/2.
    """
    half_base = float(fwhm_nm)  # base = 2*fwhm ⇒ half_base=fwhm
    x = np.abs(lambdas_nm - center_nm)
    tri = np.clip(1.0 - x / half_base, a_min=0.0, a_max=1.0)
    return tri


def box_srf(lambdas_nm: NDArrayF, center_nm: float, fwhm_nm: float) -> NDArrayF:
    """
    Rectangular (top-hat) SRF with width equal to FWHM.
    """
    half = float(fwhm_nm) / 2.0
    return ((lambdas_nm >= center_nm - half) & (lambdas_nm <= center_nm + half)).astype(np.float64)


def skewed_gaussian_srf(lambdas_nm: NDArrayF, center_nm: float, fwhm_nm: float, skew: float = 0.0) -> NDArrayF:
    """
    Skewed Gaussian using two different sigmas left/right. skew in [-0.9, 0.9].
    Effective FWHM near requested; minor deviations acceptable for augmentation.
    """
    skew = float(np.clip(skew, -0.9, 0.9))
    base_sigma = float(fwhm_nm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_l = base_sigma * (1.0 + skew)
    sigma_r = base_sigma * (1.0 - skew)
    out = np.zeros_like(lambdas_nm, dtype=np.float64)
    left = lambdas_nm <= center_nm
    right = ~left
    out[left] = np.exp(-0.5 * ((lambdas_nm[left] - center_nm) / sigma_l) ** 2)
    out[right] = np.exp(-0.5 * ((lambdas_nm[right] - center_nm) / sigma_r) ** 2)
    return out


def normalize_srf_curve(wavelength_nm: NDArrayF, response: NDArrayF) -> NDArrayF:
    """
    Normalize SRF curve to unit area under trapezoidal rule on its own support.
    """
    if np.all(response == 0):
        return response.copy()
    area = np.trapz(response, wavelength_nm)
    if area <= 0:
        return response.copy()
    return response / area


def _interp_to_grid(curve_nm: NDArrayF, curve_resp: NDArrayF, grid: CanonicalGrid) -> NDArrayF:
    """
    Interpolate SRF curve to canonical grid using linear interpolation,
    extrapolating zeros outside support.
    """
    f = interp1d(curve_nm, curve_resp, kind="linear", bounds_error=False, fill_value=0.0, assume_sorted=True)
    return f(grid.lambdas_nm).astype(np.float64)


def build_srf_matrix_from_curves(srfs: Iterable[SRF], grid: CanonicalGrid) -> SRFMatrix:
    """
    Convert a collection of SRF curves into a normalized SRFMatrix R on the canonical grid.

    Each row r_i is non-negative and sums to 1 (row-stochastic). We interpolate each SRF
    onto the grid, then renormalize w.r.t. the grid step.
    """
    srfs_list = list(srfs)
    if len(srfs_list) == 0:
        raise ValueError("No SRFs provided")

    # interpolate and normalize on the canonical grid
    R_rows: List[np.ndarray] = []
    centers = []
    fwhms = []
    for s in srfs_list:
        resp_norm = normalize_srf_curve(s.wavelength_nm, s.response)
        r = _interp_to_grid(s.wavelength_nm, resp_norm, grid)
        # Convert to discrete weights; proportional to response * Δλ (uniform grid ⇒ Δ cancels after renorm)
        r = np.maximum(r, 0.0)
        ssum = np.sum(r)
        if ssum <= 0:
            # Degenerate SRF curve; fall back to near-delta at nearest grid point to center
            r = np.zeros_like(grid.lambdas_nm)
            c = s.center_nm or float(np.mean(s.wavelength_nm))
            idx = int(np.argmin(np.abs(grid.lambdas_nm - c)))
            r[idx] = 1.0
        else:
            r /= ssum
        R_rows.append(r)
        centers.append(s.center_nm if s.center_nm is not None else float(np.sum(grid.lambdas_nm * r)))
        fwhms.append(s.fwhm_nm if s.fwhm_nm is not None else float(1.0 / np.sum(r**2) * grid.step_nm))  # approx

    R = np.vstack(R_rows)  # [N,K]
    # hash includes grid + R contents
    sha = sha256_ndarray(R)
    ew_bins = 1.0 / np.maximum(np.sum(R**2, axis=1), 1e-12)

    return SRFMatrix(
        R=R,
        centers_nm=np.asarray(centers, dtype=np.float64),
        fwhm_nm=np.asarray(fwhms, dtype=np.float64),
        effective_width_bins_=ew_bins,
        grid_name=grid.name,
        sha256=sha,
        meta={},
    )


def project_srf_matrix_onto_basis(R: SRFMatrix, proj_basis: SRFProjectionBasis) -> np.ndarray:
    """
    Project each SRF row r_i[K] onto an orthonormal basis S[K,Ms] to obtain low-dim shape coefficients.
    Returns s_coefs [N, Ms] with:
        s_i = r_i^T S, since columns of S are orthonormal under trapezoid inner product.
    """
    if proj_basis.S.shape[0] != R.R.shape[1]:
        raise ValueError("Projection basis and SRFMatrix grid dimension mismatch")
    return R.R @ proj_basis.S  # [N,Ms]
