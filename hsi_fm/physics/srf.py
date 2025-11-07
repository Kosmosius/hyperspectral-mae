"""Spectral response function utilities implemented with PyTorch tensors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import math

import torch


Tensor = torch.Tensor


@dataclass
class SpectralResponseFunction:
    """Represents a per-band spectral response function."""

    centers: Sequence[float]
    widths: Sequence[float]
    responses: Optional[Tensor] = None

    def __post_init__(self) -> None:
        if len(self.centers) != len(self.widths):
            raise ValueError("centers and widths must have the same length")
        if self.responses is not None and self.responses.shape[0] != len(self.centers):
            raise ValueError("responses must have shape (bands, samples)")


def gaussian_srf_torch(
    centers_nm: Tensor,
    fwhm_nm: Tensor,
    wavelengths_nm: Tensor,
) -> Tensor:
    """Create Gaussian SRFs on a dense wavelength grid."""

    if centers_nm.shape != fwhm_nm.shape:
        raise ValueError("centers and fwhm must have equal shape")
    sigma = fwhm_nm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    diff = wavelengths_nm.unsqueeze(0) - centers_nm.unsqueeze(-1)
    weights = torch.exp(-0.5 * (diff / sigma.unsqueeze(-1)) ** 2)
    weights_sum = weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return weights / weights_sum


def normalise_srf(srf: Tensor) -> Tensor:
    total = srf.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return srf / total


def convolve_srf_torch(hi_spec: Tensor, hi_wvl: Tensor, srf: Tensor) -> Tensor:
    """Convolve a high-resolution spectrum with the provided SRF."""

    if hi_spec.ndim == 1:
        hi_spec = hi_spec.unsqueeze(0)
    if hi_spec.ndim != 2:
        raise ValueError("hi_spec must be (B, L)")
    if hi_wvl.ndim != 1:
        raise ValueError("hi_wvl must be 1D")
    if srf.ndim != 2:
        raise ValueError("srf must be 2D")
    if hi_spec.shape[1] != hi_wvl.shape[0]:
        raise ValueError("Spectrum and wavelength grid mismatch")
    if srf.shape[1] != hi_wvl.shape[0]:
        raise ValueError("SRF grid mismatch")
    srf_norm = normalise_srf(srf)
    return hi_spec @ srf_norm.transpose(0, 1)


def gaussian_srf(
    centers: Sequence[float], fwhm: Sequence[float], wavelengths: Sequence[float]
):
    centers_t = torch.tensor(centers, dtype=torch.float64)
    fwhm_t = torch.tensor(fwhm, dtype=torch.float64)
    wavelengths_t = torch.tensor(wavelengths, dtype=torch.float64)
    return gaussian_srf_torch(centers_t, fwhm_t, wavelengths_t).tolist()


def convolve_srf(
    spectra: Sequence[Sequence[float]],
    wavelengths: Sequence[float],
    srf: SpectralResponseFunction,
):
    spectra_t = torch.tensor(spectra, dtype=torch.float64)
    wavelengths_t = torch.tensor(wavelengths, dtype=torch.float64)
    if srf.responses is None:
        srf_t = torch.tensor(
            gaussian_srf(srf.centers, srf.widths, wavelengths), dtype=torch.float64
        )
    else:
        srf_t = torch.as_tensor(srf.responses, dtype=torch.float64)
    return convolve_srf_torch(spectra_t, wavelengths_t, srf_t).tolist()


__all__ = [
    "SpectralResponseFunction",
    "gaussian_srf",
    "gaussian_srf_torch",
    "convolve_srf",
    "convolve_srf_torch",
    "normalise_srf",
]
