"""Unit tests for SRF convolution."""
from __future__ import annotations

import math

from hsi_fm.physics.srf import SpectralResponseFunction, convolve_srf, gaussian_srf


def _linspace(start: float, stop: float, num: int) -> list[float]:
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def _assert_close(a: list[list[float]], b: list[list[float]], tol: float = 1e-6) -> None:
    for row_a, row_b in zip(a, b):
        for val_a, val_b in zip(row_a, row_b):
            assert abs(val_a - val_b) <= tol


def test_gaussian_srf_normalization() -> None:
    centers = [1.0, 1.1]
    fwhm = [0.02, 0.02]
    wavelengths = _linspace(0.9, 1.2, 100)
    srf = gaussian_srf(centers, fwhm, wavelengths)
    for row in srf:
        assert abs(sum(row) - 1.0) < 1e-6


def test_convolve_srf_matches_manual_sum() -> None:
    wavelengths = _linspace(0.9, 1.2, 100)
    spectra = [
        [math.sin(w * 5) for w in wavelengths],
        [math.cos(w * 3) for w in wavelengths],
    ]
    srf = SpectralResponseFunction(centers=[1.0, 1.1], widths=[0.05, 0.05])
    result = convolve_srf(spectra, wavelengths, srf)
    responses = gaussian_srf(srf.centers, srf.widths, wavelengths)
    manual = []
    for spectrum in spectra:
        manual.append([sum(s * r for s, r in zip(spectrum, response)) for response in responses])
    _assert_close(result, manual)
