import math

import torch

from hsi_fm.physics.srf import convolve_srf_torch, gaussian_srf_torch, normalise_srf


def test_gaussian_variance_addition() -> None:
    centers = torch.tensor([-1.0, 0.0, 1.0])
    wavelengths = torch.linspace(-5.0, 5.0, 2048)
    sigma_spec = 0.5
    sigma_srf = 0.75
    fwhm = torch.full_like(centers, 2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma_srf)
    spectrum = torch.exp(-0.5 * (wavelengths / sigma_spec) ** 2)
    srf = gaussian_srf_torch(centers, fwhm, wavelengths)
    convolved = convolve_srf_torch(spectrum, wavelengths, srf)[0]
    sigma_combined = math.sqrt(sigma_spec**2 + sigma_srf**2)
    scale = sigma_spec / sigma_combined
    expected = scale * torch.exp(-0.5 * (centers / sigma_combined) ** 2)
    assert torch.allclose(convolved, expected, atol=1e-2)


def test_row_normalisation() -> None:
    centers = torch.tensor([400.0, 500.0])
    fwhm = torch.tensor([10.0, 12.0])
    wavelengths = torch.linspace(350.0, 550.0, 128)
    srf = gaussian_srf_torch(centers, fwhm, wavelengths)
    sums = srf.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-3)


def test_batched_convolution_shapes() -> None:
    wavelengths = torch.linspace(400.0, 700.0, 256)
    centers = torch.tensor([450.0, 550.0, 650.0])
    fwhm = torch.full((3,), 8.0)
    srf = normalise_srf(torch.rand(3, wavelengths.shape[0]))
    spectra = torch.rand(4, wavelengths.shape[0])
    result = convolve_srf_torch(spectra, wavelengths, srf)
    assert result.shape == (4, 3)
