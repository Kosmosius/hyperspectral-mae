"""Physics adapters bridging lab and overhead domains."""

from hsi_fm.physics.renderer import LabToSensorRenderer
from hsi_fm.physics.srf import (
    SpectralResponseFunction,
    convolve_srf,
    convolve_srf_torch,
    gaussian_srf,
    gaussian_srf_torch,
    normalise_srf,
)

__all__ = [
    "SpectralResponseFunction",
    "gaussian_srf",
    "convolve_srf",
    "LabToSensorRenderer",
    "gaussian_srf_torch",
    "convolve_srf_torch",
    "normalise_srf",
]
