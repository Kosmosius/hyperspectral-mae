"""Unit tests for SRF utilities."""
from __future__ import annotations

import math

import torch
from collections import OrderedDict

from hsi_fm.physics.srf import SRF, canonical_grid, convolve_srf, gaussian_srf


def test_gaussian_normalization() -> None:
    grid = canonical_grid(900.0, 1000.0, 1.0)
    centers = torch.tensor([930.0, 960.0, 990.0])
    fwhm = torch.tensor([10.0, 12.0, 14.0])
    matrix = gaussian_srf(centers, fwhm, grid)
    sums = matrix.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-3)


def test_gaussian_on_gaussian_variance() -> None:
    grid = canonical_grid(950.0, 1050.0, 0.5)
    centers = grid
    sigma_signal = 5.0
    sigma_srf = 8.0
    fwhm = torch.full_like(centers, sigma_srf * math.sqrt(8.0 * math.log(2.0)))
    srf_matrix = gaussian_srf(centers, fwhm, grid)
    signal = torch.exp(-0.5 * ((grid - grid.mean()) / sigma_signal) ** 2)
    convolved = convolve_srf(signal.unsqueeze(0), grid, srf_matrix).squeeze(0)
    weights = convolved / convolved.sum()
    mean = (weights * centers).sum()
    variance = ((centers - mean) ** 2 * weights).sum().item()
    expected = sigma_signal**2 + sigma_srf**2
    assert abs(variance - expected) / expected < 0.1


def test_srf_class_builds_cached_matrix() -> None:
    grid = canonical_grid(900.0, 920.0, 1.0)
    centers = torch.tensor([905.0, 910.0, 915.0])
    fwhm = torch.tensor([8.0, 8.0, 8.0])
    srf = SRF(centers_nm=centers, fwhm_nm=fwhm)
    cache = OrderedDict()
    matrix, pinv = srf.build_matrix(grid, cache=cache)
    assert matrix.shape == (3, grid.numel())
    assert pinv.shape[0] == grid.numel()
    matrix2, _ = srf.build_matrix(grid, cache=cache)
    assert torch.equal(matrix, matrix2)
