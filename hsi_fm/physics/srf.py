"""Spectral response function utilities implemented with pure Python lists."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import math


Number = float


@dataclass
class SpectralResponseFunction:
    """Represents a per-band spectral response function."""

    centers: Sequence[Number]
    widths: Sequence[Number]
    responses: Optional[List[List[Number]]] = None

    def __post_init__(self) -> None:
        if len(self.centers) != len(self.widths):
            raise ValueError("centers and widths must have the same length")
        if self.responses is not None and len(self.responses) != len(self.centers):
            raise ValueError("responses must have shape (bands, samples)")


def gaussian_srf(
    centers: Sequence[Number], fwhm: Sequence[Number], wavelengths: Sequence[Number]
) -> List[List[Number]]:
    """Create normalized Gaussian SRFs."""

    if len(centers) != len(fwhm):
        raise ValueError("centers and fwhm must have equal length")
    result: List[List[Number]] = []
    two_ln2 = 2.0 * math.log(2.0)
    for center, width in zip(centers, fwhm):
        sigma = width / (2.0 * math.sqrt(two_ln2))
        weights: List[Number] = []
        for wavelength in wavelengths:
            diff = (wavelength - center) / sigma
            weights.append(math.exp(-0.5 * diff * diff))
        total = sum(weights) + 1e-12
        result.append([w / total for w in weights])
    return result


def convolve_srf(
    spectra: Sequence[Sequence[Number]],
    wavelengths: Sequence[Number],
    srf: SpectralResponseFunction,
) -> List[List[Number]]:
    """Convolve high-resolution spectra with a spectral response function."""

    spectra_list = [list(row) for row in spectra]
    if not spectra_list:
        return []
    if len(spectra_list[0]) != len(wavelengths):
        raise ValueError("spectra and wavelengths size mismatch")
    if srf.responses is None:
        responses = gaussian_srf(srf.centers, srf.widths, wavelengths)
    else:
        responses = [row[:] for row in srf.responses]
        for row in responses:
            total = sum(row) + 1e-12
            for i, val in enumerate(row):
                row[i] = val / total
    if len(responses[0]) != len(wavelengths):
        raise ValueError("SRF response grid mismatch")

    result: List[List[Number]] = []
    for spectrum in spectra_list:
        row: List[Number] = []
        for response in responses:
            value = sum(s * r for s, r in zip(spectrum, response))
            row.append(value)
        result.append(row)
    return result


__all__ = ["SpectralResponseFunction", "gaussian_srf", "convolve_srf"]
