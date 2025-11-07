"""Physics adapters bridging lab and overhead domains."""

from hsi_fm.physics.renderer import LabToSensorRenderer
from hsi_fm.physics.srf import SpectralResponseFunction, convolve_srf, gaussian_srf

__all__ = [
    "SpectralResponseFunction",
    "gaussian_srf",
    "convolve_srf",
    "LabToSensorRenderer",
]
