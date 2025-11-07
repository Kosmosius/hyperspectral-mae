"""Physics adapters bridging lab and overhead domains."""

from .srf import gaussian_srf, convolve_srf
from .renderer import LabToSensorRenderer

__all__ = ["gaussian_srf", "convolve_srf", "LabToSensorRenderer"]
