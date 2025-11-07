# hsi_fm/tests/test_renderer.py
"""Tests for the lab-to-sensor renderer."""
from __future__ import annotations

import math
from typing import List

from hsi_fm.physics.renderer import AtmosphericState, LabToSensorRenderer
from hsi_fm.physics.srf import SpectralResponseFunction


def _linspace(start: float, stop: float, num: int) -> list[float]:
    if num <= 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def _jittered_grid(start: float, stop: float, num: int, frac: float = 0.2) -> list[float]:
    """
    Build a slightly non-uniform, but strictly increasing, grid by adding a
    small sinusoidal perturbation then sorting. 'frac' scales the perturbation
    relative to the uniform step (keep <= 0.5 to preserve order after sorting).
    """
    base = _linspace(start, stop, num)
    if num <= 2 or frac <= 0.0:
        return base
    step = (stop - start) / (num - 1)
    out = []
    for i, x in enumerate(base):
        # bounded perturbation; deterministic (no RNG)
        dx = frac * 0.3 * step * math.sin(2.0 * math.pi * i / 13.0)
        out.append(x + dx)
    out.sort()
    return out


def _assert_close_matrix(a: list[list[float]], b: list[list[float]], tol: float = 5e-3) -> None:
    assert len(a) == len(b), f"Row count mismatch: {len(a)} vs {len(b)}"
    for r, (row_a, row_b) in enumerate(zip(a, b)):
        assert len(row_a) == len(row_b), f"Col count mismatch at row {r}: {len(row_a)} vs {len(row_b)}"
        for c, (va, vb) in enumerate(zip(row_a, row_b)):
            diff = abs(va - vb)
            assert diff <= tol, f"Mismatch at [{r},{c}] |a-b|={diff:.6g} > {tol}"


def _make_simple_ramps(L: int) -> list[list[float]]:
    # Two smooth reflectance ramps in [0,1] to keep the round-trip well-conditioned
    r1 = _linspace(0.10, 0.30, L)
    r2 = _linspace(0.20, 0.40, L)
    return [r1, r2]


def test_renderer_round_trip_consistency_uniform_grid() -> None:
    # Uniform grid
    wavelengths = _linspace(0.9, 1.1, 121)  # 121 points over 200 nm span (arbitrary units consistent with widths)
    # Use more than 3 bands to improve the projection quality in the bandspace
    centers = _linspace(0.92, 1.08, 7)      # 7 Gaussian bands
    widths = [0.02] * len(centers)          # FWHM per band

    srf = SpectralResponseFunction(centers=centers, widths=widths)
    renderer = LabToSensorRenderer(srf)

    reflectance = _make_simple_ramps(len(wavelengths))
    atmosphere = AtmosphericState(irradiance=1.2, transmittance=0.8, path_radiance=0.01)

    # Forward: R_hi -> L_sensor
    radiance = renderer.render(reflectance, wavelengths, atmosphere)

    # Invert with a small ridge for numerical stability, then re-render
    recovered = renderer.invert(radiance, wavelengths, atmosphere, ridge=1e-6)
    rerendered = renderer.render(recovered, wavelengths, atmosphere)

    # Round-trip should be very close in bandspace
    _assert_close_matrix(radiance, rerendered, tol=5e-3)


def test_renderer_round_trip_consistency_nonu_grid() -> None:
    # Mildly non-uniform grid to exercise trapezoidal SRF normalization
    wavelengths = _jittered_grid(0.9, 1.1, 121, frac=0.2)
    centers = _linspace(0.92, 1.08, 7)
    widths = [0.02] * len(centers)

    srf = SpectralResponseFunction(centers=centers, widths=widths)
    renderer = LabToSensorRenderer(srf)

    reflectance = _make_simple_ramps(len(wavelengths))
    atmosphere = AtmosphericState(irradiance=1.1, transmittance=0.85, path_radiance=0.005)

    radiance = renderer.render(reflectance, wavelengths, atmosphere)
    recovered = renderer.invert(radiance, wavelengths, atmosphere, ridge=1e-6)
    rerendered = renderer.render(recovered, wavelengths, atmosphere)

    # Non-uniform grids are a bit tougher; allow a slightly looser tolerance
    _assert_close_matrix(radiance, rerendered, tol=1e-2)


def test_renderer_handles_single_spectrum() -> None:
    wavelengths = _linspace(0.9, 1.1, 50)
    centers = [0.95, 1.0, 1.05]
    widths = [0.02, 0.02, 0.02]
    srf = SpectralResponseFunction(centers=centers, widths=widths)
    renderer = LabToSensorRenderer(srf)

    reflectance = _linspace(0.1, 0.3, 50)
    radiance = renderer.render(reflectance, wavelengths)

    assert isinstance(radiance, list)
    assert len(radiance) == 1
    assert len(radiance[0]) == len(centers)
