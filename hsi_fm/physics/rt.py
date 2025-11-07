"""Mock radiative transfer helpers for deterministic testing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

__all__ = [
    "SWIRRTConfig",
    "LWIRRTConfig",
    "swir_reflectance_to_radiance",
    "swir_radiance_to_reflectance",
    "lwir_emit_radiance",
]


@dataclass
class SWIRRTConfig:
    """Configuration for the SWIR mock RT."""

    tau: float = 0.9
    scale: float = 1.0
    noise_std: float = 0.0


@dataclass
class LWIRRTConfig:
    """Configuration for the LWIR mock RT."""

    gain: float = 1e-3
    offset: float = 1.0


def _deterministic_noise(shape: torch.Size, std: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    if std <= 0:
        return torch.zeros(shape, device=device, dtype=dtype)
    idx = torch.arange(shape[-1], device=device, dtype=dtype)
    base = torch.sin(idx * 0.1) + torch.cos(idx * 0.05)
    while base.ndim < len(shape):
        base = base.unsqueeze(0)
    return std * base.expand(shape)


def swir_reflectance_to_radiance(
    reflectance: Tensor,
    solar_irradiance: Tensor,
    tau: float,
    scale: float,
    noise_std: float = 0.0,
) -> Tensor:
    """Convert reflectance to radiance using a linear mock RT."""

    if reflectance.ndim != 2:
        raise ValueError("reflectance must be (batch, L)")
    if solar_irradiance.ndim != 1:
        raise ValueError("solar_irradiance must be 1-D")
    if reflectance.shape[1] != solar_irradiance.shape[0]:
        raise ValueError("Spectrum and irradiance mismatch")
    device = reflectance.device
    dtype = reflectance.dtype
    irradiance = solar_irradiance.to(device=device, dtype=dtype)
    radiance = scale * tau * reflectance * irradiance
    noise = _deterministic_noise(radiance.shape, noise_std, device, dtype)
    return (radiance + noise).clamp_min(0.0)


def swir_radiance_to_reflectance(
    radiance: Tensor,
    solar_irradiance: Tensor,
    tau: float,
    scale: float,
) -> Tensor:
    """Invert the mock SWIR transform."""

    if radiance.ndim != 2:
        raise ValueError("radiance must be (batch, L)")
    if solar_irradiance.ndim != 1:
        raise ValueError("solar_irradiance must be 1-D")
    if radiance.shape[1] != solar_irradiance.shape[0]:
        raise ValueError("Spectrum and irradiance mismatch")
    denom = (scale * tau * solar_irradiance.to(device=radiance.device, dtype=radiance.dtype)).clamp_min(1e-6)
    reflectance = radiance / denom
    return reflectance.clamp(0.0, 1.0)


def lwir_emit_radiance(
    temperature: Tensor,
    emissivity: Tensor,
    planck_lut: Tensor,
    *,
    gain: float,
    offset: float,
) -> Tensor:
    """Generate LWIR radiance with a differentiable surrogate."""

    if temperature.ndim != 1:
        raise ValueError("temperature must be (batch,)")
    if emissivity.ndim != 2:
        raise ValueError("emissivity must be (batch, L)")
    if planck_lut.ndim != 1:
        raise ValueError("planck_lut must be 1-D")
    if emissivity.shape[1] != planck_lut.shape[0]:
        raise ValueError("emissivity and planck_lut mismatch")
    if emissivity.shape[0] != temperature.shape[0]:
        raise ValueError("Batch size mismatch")
    temp_term = gain * temperature.to(dtype=emissivity.dtype, device=emissivity.device).unsqueeze(-1) + offset
    planck = planck_lut.to(device=emissivity.device, dtype=emissivity.dtype)
    radiance = emissivity * planck * temp_term
    return radiance.clamp_min(0.0)
