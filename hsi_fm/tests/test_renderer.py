"""Tests for the lab-to-sensor renderer."""
from __future__ import annotations

from hsi_fm.physics.renderer import AtmosphericState, LabToSensorRenderer
from hsi_fm.physics.srf import SpectralResponseFunction


def _linspace(start: float, stop: float, num: int) -> list[float]:
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def _assert_close(a: list[list[float]], b: list[list[float]], tol: float = 5e-2) -> None:
    for row_a, row_b in zip(a, b):
        for val_a, val_b in zip(row_a, row_b):
            assert abs(val_a - val_b) <= tol


def test_renderer_round_trip_consistency() -> None:
    wavelengths = _linspace(0.9, 1.1, 50)
    centers = [0.95, 1.0, 1.05]
    widths = [0.02, 0.02, 0.02]
    srf = SpectralResponseFunction(centers=centers, widths=widths)
    renderer = LabToSensorRenderer(srf)
    reflectance = [_linspace(0.1, 0.3, 50), _linspace(0.2, 0.4, 50)]
    atmosphere = AtmosphericState(irradiance=1.2, transmittance=0.8, path_radiance=0.01)
    radiance = renderer.render(reflectance, wavelengths, atmosphere)
    recovered = renderer.invert(radiance, wavelengths, atmosphere)
    rerendered = renderer.render(recovered, wavelengths, atmosphere)
    _assert_close(radiance, rerendered, tol=1e-3)


def test_renderer_handles_single_spectrum() -> None:
    wavelengths = _linspace(0.9, 1.1, 50)
    centers = [0.95, 1.0, 1.05]
    widths = [0.02, 0.02, 0.02]
    srf = SpectralResponseFunction(centers=centers, widths=widths)
    renderer = LabToSensorRenderer(srf)
    reflectance = _linspace(0.1, 0.3, 50)
    radiance = renderer.render(reflectance, wavelengths)
    assert len(radiance) == 1
    assert len(radiance[0]) == len(centers)

