"""Tests for the lab-to-sensor renderer."""
from __future__ import annotations

import torch

from hsi_fm.physics.renderer import LabToSensorRenderer
from hsi_fm.physics.srf import canonical_grid


def _make_renderer() -> LabToSensorRenderer:
    grid = canonical_grid(1000.0, 1100.0, 1.0)
    centers = torch.linspace(1000.0, 1100.0, 16)
    fwhm = torch.full((16,), 12.0)
    return LabToSensorRenderer(
        hi_wvl_nm=grid,
        sensor_centers_nm=centers,
        sensor_fwhm_nm=fwhm,
        mode="swir_reflectance_to_radiance",
        rt_params={"tau": 0.9, "scale": 1.05, "noise_std": 0.0},
    )


def test_round_trip_reflectance() -> None:
    renderer = _make_renderer()
    grid = renderer.hi_wvl_nm
    reflectance = torch.stack(
        [0.3 + 0.1 * torch.sin(grid / 50.0), 0.4 + 0.05 * torch.cos(grid / 40.0)],
        dim=0,
    ).clamp(0.05, 0.95)
    sensor, _ = renderer.render(reflectance)
    recovered, _ = renderer.inverse(sensor)
    assert torch.mean((reflectance - recovered) ** 2).item() < 5e-4


def test_renderer_cache_reuse() -> None:
    renderer = _make_renderer()
    reflectance = torch.rand(1, renderer.hi_wvl_nm.numel()) * 0.5 + 0.25
    renderer.render(reflectance)
    assert len(renderer._cache) == 1  # type: ignore[attr-defined]
    first_matrix = next(iter(renderer._cache.values()))[0]  # type: ignore[attr-defined]
    renderer.render(reflectance)
    second_matrix = next(iter(renderer._cache.values()))[0]  # type: ignore[attr-defined]
    assert first_matrix.data_ptr() == second_matrix.data_ptr()
