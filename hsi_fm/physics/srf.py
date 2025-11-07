"""Spectral response function helpers implemented with PyTorch."""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

__all__ = ["SRF", "gaussian_srf", "convolve_srf", "canonical_grid"]

_EPS = 1e-12
_ROUND = 1e6
_SQRT8LN2 = math.sqrt(8.0 * math.log(2.0))


@dataclass
class SRF:
    """Container describing a sensor's spectral response function.

    Parameters
    ----------
    centers_nm:
        1-D tensor of band center wavelengths in nanometres.
    fwhm_nm:
        Full-width half-maximum per band. Required when ``responses`` is ``None``.
    responses:
        Optional pre-tabulated SRF samples shaped ``(bands, hi_samples)``.
    """

    centers_nm: Tensor
    fwhm_nm: Optional[Tensor] = None
    responses: Optional[Tensor] = None

    def __post_init__(self) -> None:
        self.centers_nm = torch.as_tensor(self.centers_nm, dtype=torch.float64)
        if self.centers_nm.ndim != 1:
            raise ValueError("centers_nm must be 1-D")
        if self.responses is not None and self.fwhm_nm is not None:
            raise ValueError("Specify either fwhm_nm or responses, not both")
        if self.responses is None and self.fwhm_nm is None:
            raise ValueError("Either fwhm_nm or responses must be provided")
        if self.fwhm_nm is not None:
            self.fwhm_nm = torch.as_tensor(self.fwhm_nm, dtype=torch.float64)
            if self.fwhm_nm.shape != self.centers_nm.shape:
                raise ValueError("fwhm_nm must match centers_nm shape")
        if self.responses is not None:
            self.responses = torch.as_tensor(self.responses, dtype=torch.float64)
            if self.responses.ndim != 2:
                raise ValueError("responses must be 2-D")
            if self.responses.shape[0] != self.centers_nm.shape[0]:
                raise ValueError("responses first dim must equal number of bands")

    def build_matrix(
        self,
        hi_wvl_nm: Tensor,
        *,
        normalize: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        cache: Optional[OrderedDict[tuple, tuple[Tensor, Tensor]]] = None,
    ) -> tuple[Tensor, Tensor]:
        """Return SRF matrix and its pseudoinverse on the requested grid."""

        hi_wvl_nm = torch.as_tensor(hi_wvl_nm, dtype=torch.float64)
        if hi_wvl_nm.ndim != 1:
            raise ValueError("hi_wvl_nm must be 1-D")

        key = None
        if cache is not None:
            key = (
                tuple(torch.round(self.centers_nm * _ROUND).tolist()),
                tuple(
                    []
                    if self.fwhm_nm is None
                    else torch.round(self.fwhm_nm * _ROUND).tolist()
                ),
                tuple(torch.round(hi_wvl_nm * _ROUND).tolist()),
                normalize,
            )
            if key in cache:
                matrix, pinv = cache.pop(key)
                cache[key] = (matrix, pinv)
                return (
                    matrix.to(device=device, dtype=dtype) if dtype or device else matrix,
                    pinv.to(device=device, dtype=dtype) if dtype or device else pinv,
                )

        if self.responses is not None:
            matrix = self.responses
            if matrix.shape[1] != hi_wvl_nm.shape[0]:
                raise ValueError("Tabulated SRF grid does not match hi_wvl_nm")
        else:
            matrix = gaussian_srf(self.centers_nm, self.fwhm_nm, hi_wvl_nm, normalize=False)

        if normalize:
            matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(_EPS)

        target_dtype = dtype or torch.float32
        matrix = matrix.to(dtype=target_dtype, device=device)
        pinv = torch.linalg.pinv(matrix).to(dtype=target_dtype, device=device)

        if cache is not None and key is not None:
            cache[key] = (matrix.detach(), pinv.detach())
            if len(cache) > 16:
                cache.popitem(last=False)

        return matrix, pinv


def gaussian_srf(
    centers_nm: Tensor | float | list[float],
    fwhm_nm: Tensor | float | list[float],
    hi_wvl_nm: Tensor | float | list[float],
    *,
    normalize: bool = True,
) -> Tensor:
    """Sample Gaussian SRFs on the provided high-resolution grid."""

    centers = torch.as_tensor(centers_nm, dtype=torch.float64)
    fwhm = torch.as_tensor(fwhm_nm, dtype=torch.float64)
    hi = torch.as_tensor(hi_wvl_nm, dtype=torch.float64)
    if centers.ndim != 1 or fwhm.ndim != 1:
        raise ValueError("centers and fwhm must be 1-D")
    if centers.shape != fwhm.shape:
        raise ValueError("centers and fwhm must have identical shapes")
    if hi.ndim != 1:
        raise ValueError("hi_wvl_nm must be 1-D")
    sigma = (fwhm / _SQRT8LN2).clamp_min(1e-6)
    diff = hi.unsqueeze(0) - centers.unsqueeze(1)
    weights = torch.exp(-0.5 * (diff / sigma.unsqueeze(1)) ** 2)
    if normalize:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(_EPS)
    return weights.to(dtype=torch.float32)


def convolve_srf(hi_spec: Tensor, hi_wvl_nm: Tensor, srf_matrix: Tensor) -> Tensor:
    """Convolve high-resolution spectra with an SRF matrix."""

    if hi_spec.ndim != 2:
        raise ValueError("hi_spec must be (batch, hi_samples)")
    if hi_wvl_nm.ndim != 1:
        raise ValueError("hi_wvl_nm must be 1-D")
    if srf_matrix.ndim != 2:
        raise ValueError("srf_matrix must be 2-D")
    if hi_spec.shape[1] != hi_wvl_nm.shape[0]:
        raise ValueError("Spectrum and wavelength grid mismatch")
    if srf_matrix.shape[1] != hi_wvl_nm.shape[0]:
        raise ValueError("SRF matrix grid mismatch")
    return hi_spec @ srf_matrix.transpose(-1, -2)


def canonical_grid(min_nm: float, max_nm: float, step_nm: float) -> Tensor:
    """Create a canonical wavelength grid inclusive of the upper bound."""

    if step_nm <= 0:
        raise ValueError("step_nm must be positive")
    max_inclusive = max_nm + 0.5 * step_nm
    grid = torch.arange(min_nm, max_inclusive, step_nm, dtype=torch.float32)
    if grid[-1] > max_nm + 1e-6:
        grid = grid[:-1]
    return grid
