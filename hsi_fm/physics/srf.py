# hsi_fm/physics/srf.py
"""
Spectral response function (SRF) utilities implemented with PyTorch tensors.

Design goals
------------
- Works on CPU/GPU with proper dtype/device propagation.
- Supports both Gaussian-generated SRFs and tabulated SRFs.
- Correct row normalization on *non-uniform* wavelength grids via trapezoidal rules.
- Batched, vectorized convolution: (B, L) @ (L, K) -> (B, K).
- Backward compatibility with existing public names:
  - SpectralResponseFunction (dataclass)
  - gaussian_srf, gaussian_srf_torch
  - convolve_srf, convolve_srf_torch
  - normalise_srf (British spelling kept)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Iterable, Tuple

import math
import torch

Tensor = torch.Tensor

__all__ = [
    "SpectralResponseFunction",
    "fwhm_to_sigma",
    "sigma_to_fwhm",
    "trapezoid_weights",
    "gaussian_srf_torch",
    "tabulated_srf_torch",
    "normalise_srf",
    "convolve_srf_torch",
    # Back-compat Python-list wrappers:
    "gaussian_srf",
    "convolve_srf",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fwhm_to_sigma(fwhm: Tensor) -> Tensor:
    """Convert FWHM to sigma for a Gaussian: σ = FWHM / (2√(2 ln 2))."""
    return fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def sigma_to_fwhm(sigma: Tensor) -> Tensor:
    """Convert sigma to FWHM for a Gaussian: FWHM = 2√(2 ln 2) · σ."""
    return sigma * (2.0 * math.sqrt(2.0 * math.log(2.0)))


def _ensure_1d_tensor(x: Tensor | Sequence[float], *, dtype: torch.dtype, device: torch.device) -> Tensor:
    if isinstance(x, torch.Tensor):
        t = x
        if t.ndim != 1:
            raise ValueError("Expected 1D tensor.")
        return t.to(dtype=dtype, device=device)
    t = torch.tensor(x, dtype=dtype, device=device)
    if t.ndim != 1:
        raise ValueError("Expected 1D sequence.")
    return t


def trapezoid_weights(wavelengths_nm: Tensor) -> Tensor:
    """
    Compute trapezoidal integration weights for a (possibly non-uniform) 1D grid.
    Returns w (L,) such that sum(w * f) approximates ∫ f dλ.

    For uniform grids this reduces to a constant step with half-weights at the ends.
    """
    if wavelengths_nm.ndim != 1:
        raise ValueError("wavelengths_nm must be 1D.")
    L = wavelengths_nm.shape[0]
    if L < 2:
        return torch.ones_like(wavelengths_nm)

    # interior weights: 0.5*(Δ_{i} + Δ_{i+1})
    # endpoints: 0.5*Δ for first and last
    diffs = torch.diff(wavelengths_nm)  # (L-1,)
    w = torch.empty_like(wavelengths_nm)
    w[0] = 0.5 * diffs[0]
    w[-1] = 0.5 * diffs[-1]
    if L > 2:
        w[1:-1] = 0.5 * (diffs[:-1] + diffs[1:])
    return w.clamp_min(1e-12)


def _row_normalize(matrix: Tensor, weights: Optional[Tensor]) -> Tensor:
    """
    Normalize each row of `matrix` to unit area.
    If `weights` is provided (L,), use trapezoidal weighted sum per row.
    Otherwise, normalize by simple sum (assumes uniform grid).
    """
    if weights is None:
        denom = matrix.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return matrix / denom
    # weight each column by the integration weight
    denom = (matrix * weights.unsqueeze(0)).sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return matrix / denom


# ---------------------------------------------------------------------------
# Dataclass for SRFs
# ---------------------------------------------------------------------------

@dataclass
class SpectralResponseFunction:
    """
    Represents a per-band spectral response function on some high-resolution
    wavelength grid.

    Parameters
    ----------
    centers : Sequence[float]
        Per-band central wavelengths (nm). Used only if `responses` is None
        and we need to synthesize Gaussians with FWHM `widths`.
    widths : Sequence[float]
        Per-band FWHM (nm). Used only if `responses` is None.
    responses : Optional[Tensor], shape (K, L)
        Tabulated SRF values on a provided high-res grid (one row per band).
        These will be normalized appropriately by `normalise_srf()` when used.
    """
    centers: Sequence[float]
    widths: Sequence[float]
    responses: Optional[Tensor] = None  # (K, L)

    def __post_init__(self) -> None:
        if len(self.centers) != len(self.widths):
            raise ValueError("centers and widths must have the same length")
        if self.responses is not None and self.responses.ndim != 2:
            raise ValueError("responses must be 2D with shape (K, L)")

    @property
    def K(self) -> int:
        return len(self.centers) if self.responses is None else int(self.responses.shape[0])


# ---------------------------------------------------------------------------
# SRF construction and normalization
# ---------------------------------------------------------------------------

def gaussian_srf_torch(
    centers_nm: Tensor,
    fwhm_nm: Tensor,
    wavelengths_nm: Tensor,
    *,
    normalize: bool = True,
) -> Tensor:
    """
    Create Gaussian SRFs on a dense wavelength grid.

    Parameters
    ----------
    centers_nm : (K,)
    fwhm_nm    : (K,) or scalar
    wavelengths_nm : (L,)
    normalize  : normalize each band to unit area on the provided grid.

    Returns
    -------
    srf : (K, L) tensor
    """
    if centers_nm.ndim != 1:
        raise ValueError("centers_nm must be 1D")
    if wavelengths_nm.ndim != 1:
        raise ValueError("wavelengths_nm must be 1D")
    if fwhm_nm.ndim == 0:
        fwhm_nm = fwhm_nm.expand_as(centers_nm)
    if centers_nm.shape != fwhm_nm.shape:
        raise ValueError("centers_nm and fwhm_nm must have the same shape")

    device = wavelengths_nm.device
    dtype = wavelengths_nm.dtype
    centers_nm = centers_nm.to(device=device, dtype=dtype)
    fwhm_nm = fwhm_nm.to(device=device, dtype=dtype)

    sigma = fwhm_to_sigma(fwhm_nm).clamp_min(torch.finfo(dtype).eps)
    # (K, L)
    x = wavelengths_nm.unsqueeze(0) - centers_nm.unsqueeze(1)
    srf = torch.exp(-0.5 * (x / sigma.unsqueeze(1)) ** 2)

    if normalize:
        w = trapezoid_weights(wavelengths_nm)
        srf = _row_normalize(srf, w)
    return srf


def tabulated_srf_torch(
    responses: Tensor,
    wavelengths_nm: Tensor,
    *,
    normalize: bool = True,
) -> Tensor:
    """
    Normalize provided tabulated SRFs to unit area on the given grid.

    Parameters
    ----------
    responses : (K, L)
    wavelengths_nm : (L,)
    """
    if responses.ndim != 2:
        raise ValueError("responses must be (K, L)")
    if wavelengths_nm.ndim != 1:
        raise ValueError("wavelengths_nm must be 1D")
    responses = responses
    if normalize:
        w = trapezoid_weights(wavelengths_nm)
        responses = _row_normalize(responses, w)
    return responses


def normalise_srf(srf: Tensor, wavelengths_nm: Optional[Tensor] = None) -> Tensor:
    """
    British-spelling alias. Normalize SRF rows to unit area. If `wavelengths_nm`
    is provided, use trapezoidal integration; otherwise use simple row-sum.
    """
    w = None if wavelengths_nm is None else trapezoid_weights(wavelengths_nm)
    return _row_normalize(srf, w)


# ---------------------------------------------------------------------------
# Convolution (spectral mixing) utilities
# ---------------------------------------------------------------------------

def convolve_srf_torch(
    hi_spec: Tensor,
    hi_wvl_nm: Tensor,
    srf: Tensor,
) -> Tensor:
    """
    Convolve a high-resolution spectrum with the provided SRF.

    Parameters
    ----------
    hi_spec : (B, L) or (L,)
        High-resolution spectra aligned with `hi_wvl_nm`.
    hi_wvl_nm : (L,)
        Wavelength grid (nm) for hi_spec and SRF's columns.
    srf : (K, L)
        Row-normalized SRF matrix on the same wavelength grid.

    Returns
    -------
    out : (B, K)
        Spectra projected into sensor bandspace.
    """
    if hi_spec.ndim == 1:
        hi_spec = hi_spec.unsqueeze(0)
    if hi_spec.ndim != 2:
        raise ValueError("hi_spec must be (B, L) or (L,)")

    if hi_wvl_nm.ndim != 1:
        raise ValueError("hi_wvl_nm must be 1D")
    if srf.ndim != 2:
        raise ValueError("srf must be 2D (K, L)")

    B, L = hi_spec.shape
    K, Ls = srf.shape
    if L != Ls:
        raise ValueError(f"Grid mismatch: hi_spec L={L}, srf L={Ls}")
    # Use matmul: (B, L) @ (L, K) = (B, K)
    return hi_spec.to(dtype=srf.dtype, device=srf.device) @ srf.transpose(0, 1).contiguous()


# ---------------------------------------------------------------------------
# Backward-compatible Python-list wrappers (useful for quick scripts/tests)
# ---------------------------------------------------------------------------

def gaussian_srf(
    centers: Sequence[float],
    fwhm: Sequence[float] | float,
    wavelengths: Sequence[float],
) -> list[list[float]]:
    device = torch.device("cpu")
    dtype = torch.float32
    centers_t = _ensure_1d_tensor(centers, dtype=dtype, device=device)
    fwhm_t = _ensure_1d_tensor(
        fwhm if isinstance(fwhm, Sequence) else [float(fwhm)] * centers_t.numel(),
        dtype=dtype,
        device=device,
    )
    wavelengths_t = _ensure_1d_tensor(wavelengths, dtype=dtype, device=device)
    srf = gaussian_srf_torch(centers_t, fwhm_t, wavelengths_t, normalize=True)
    return srf.tolist()


def convolve_srf(
    spectra: Sequence[Sequence[float]],
    wavelengths: Sequence[float],
    srf: SpectralResponseFunction,
) -> list[list[float]]:
    device = torch.device("cpu")
    dtype = torch.float32
    spectra_t = torch.tensor(spectra, dtype=dtype, device=device)
    wavelengths_t = _ensure_1d_tensor(wavelengths, dtype=dtype, device=device)

    if srf.responses is None:
        centers_t = _ensure_1d_tensor(srf.centers, dtype=dtype, device=device)
        fwhm_t = _ensure_1d_tensor(srf.widths, dtype=dtype, device=device)
        srf_t = gaussian_srf_torch(centers_t, fwhm_t, wavelengths_t, normalize=True)
    else:
        srf_t = tabulated_srf_torch(srf.responses.to(device=device, dtype=dtype), wavelengths_t, normalize=True)

    out = convolve_srf_torch(spectra_t, wavelengths_t, srf_t)
    return out.tolist()
