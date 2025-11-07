"""Renderer bridging lab spectra and sensor measurements."""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from .rt import (
    LWIRRTConfig,
    SWIRRTConfig,
    lwir_emit_radiance,
    swir_radiance_to_reflectance,
    swir_reflectance_to_radiance,
)
from .srf import SRF, canonical_grid, convolve_srf

__all__ = ["LabToSensorRenderer", "canonical_grid"]


class LabToSensorRenderer:
    """Differentiable renderer with SRF caching."""

    def __init__(
        self,
        hi_wvl_nm: Tensor,
        sensor_centers_nm: Tensor,
        sensor_fwhm_nm: Tensor,
        *,
        mode: str = "swir_reflectance_to_radiance",
        rt_params: Optional[Dict[str, float | Tensor]] = None,
        cache_size: int = 8,
    ) -> None:
        self.hi_wvl_nm = torch.as_tensor(hi_wvl_nm, dtype=torch.float32)
        if self.hi_wvl_nm.ndim != 1:
            raise ValueError("hi_wvl_nm must be 1-D")
        self.default_centers = torch.as_tensor(sensor_centers_nm, dtype=torch.float32)
        self.default_fwhm = torch.as_tensor(sensor_fwhm_nm, dtype=torch.float32)
        if self.default_centers.shape != self.default_fwhm.shape:
            raise ValueError("Sensor centers/fwhm mismatch")
        self.mode = mode
        if mode not in {"swir_reflectance", "swir_reflectance_to_radiance", "lwir_radiance"}:
            raise ValueError(f"Unsupported mode {mode}")
        self.rt_params: Dict[str, float | Tensor] = dict(rt_params or {})
        self._cache_size = cache_size
        self._cache: OrderedDict[tuple, Tuple[Tensor, Tensor]] = OrderedDict()

    def _resolve_sensor(
        self,
        centers_nm: Optional[Tensor],
        fwhm_nm: Optional[Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        centers = torch.as_tensor(
            centers_nm if centers_nm is not None else self.default_centers,
            dtype=torch.float32,
            device=device,
        )
        fwhm = torch.as_tensor(
            fwhm_nm if fwhm_nm is not None else self.default_fwhm,
            dtype=torch.float32,
            device=device,
        )
        srf = SRF(centers_nm=centers.cpu(), fwhm_nm=fwhm.cpu())
        matrix, pinv = srf.build_matrix(
            self.hi_wvl_nm.cpu(),
            dtype=dtype,
            device=device,
            cache=self._cache,
        )
        # maintain cache size
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        hi_grid = self.hi_wvl_nm.to(device=device, dtype=dtype)
        return centers, fwhm, matrix, pinv

    def _solar_irradiance(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        if "solar_irradiance" in self.rt_params:
            solar = torch.as_tensor(self.rt_params["solar_irradiance"], dtype=dtype, device=device)
            if solar.ndim != 1:
                raise ValueError("solar_irradiance must be 1-D")
            return solar
        hi = self.hi_wvl_nm.to(device=device, dtype=dtype)
        norm = (hi - hi.min()) / (hi.max() - hi.min() + 1e-6)
        return (1.0 + 0.1 * torch.sin(norm * 3.14159)).to(dtype=dtype)

    def render(
        self,
        reflectance_hi: Tensor,
        *,
        sensor_centers_nm: Optional[Tensor] = None,
        sensor_fwhm_nm: Optional[Tensor] = None,
        rt_overrides: Optional[Dict[str, float | Tensor]] = None,
        return_intermediates: bool = False,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """Render lab reflectance to sensor measurements."""

        if reflectance_hi.ndim != 2:
            raise ValueError("reflectance_hi must be (batch, L)")
        device = reflectance_hi.device
        dtype = reflectance_hi.dtype
        centers, fwhm, matrix, _ = self._resolve_sensor(
            sensor_centers_nm, sensor_fwhm_nm, dtype, device
        )
        hi_grid = self.hi_wvl_nm.to(device=device, dtype=dtype)
        params = dict(self.rt_params)
        if rt_overrides:
            params.update(rt_overrides)
        intermediates: Dict[str, Tensor] = {}

        if self.mode == "swir_reflectance":
            hi_signal = reflectance_hi
        elif self.mode == "swir_reflectance_to_radiance":
            cfg = SWIRRTConfig(
                tau=float(params.get("tau", 0.9)),
                scale=float(params.get("scale", 1.0)),
                noise_std=float(params.get("noise_std", 0.0)),
            )
            solar = params.get("solar_irradiance")
            if solar is None:
                solar = self._solar_irradiance(device, dtype)
            else:
                solar = torch.as_tensor(solar, dtype=dtype, device=device)
            hi_signal = swir_reflectance_to_radiance(
                reflectance_hi,
                solar,
                cfg.tau,
                cfg.scale,
                cfg.noise_std,
            )
            intermediates["solar_irradiance"] = solar
            intermediates["radiance_hi"] = hi_signal
            intermediates["reflectance_hi"] = reflectance_hi
        else:  # lwir_radiance
            cfg = LWIRRTConfig(
                gain=float(params.get("gain", 1e-3)),
                offset=float(params.get("offset", 1.0)),
            )
            temperature = params.get("temperature")
            if temperature is None:
                temperature = torch.full((reflectance_hi.shape[0],), 300.0, device=device, dtype=dtype)
            else:
                temperature = torch.as_tensor(temperature, dtype=dtype, device=device)
            emissivity = params.get("emissivity")
            if emissivity is None:
                emissivity = (1.0 - reflectance_hi).clamp(0.1, 1.0)
            else:
                emissivity = torch.as_tensor(emissivity, dtype=dtype, device=device)
            planck = params.get("planck_L")
            if planck is None:
                planck = torch.exp(-((hi_grid - hi_grid.mean()) ** 2) / (2 * 250.0**2))
            else:
                planck = torch.as_tensor(planck, dtype=dtype, device=device)
            hi_signal = lwir_emit_radiance(
                temperature,
                emissivity,
                planck,
                gain=cfg.gain,
                offset=cfg.offset,
            )
            intermediates["temperature"] = temperature
            intermediates["emissivity"] = emissivity
            intermediates["planck_L"] = planck
            intermediates["radiance_hi"] = hi_signal

        sensor = convolve_srf(hi_signal, hi_grid, matrix)
        intermediates["sensor_matrix"] = matrix
        intermediates["sensor_centers_nm"] = centers
        intermediates["sensor_fwhm_nm"] = fwhm
        return sensor, (intermediates if return_intermediates else None)

    def inverse(
        self,
        sensor_measurement: Tensor,
        *,
        sensor_centers_nm: Optional[Tensor] = None,
        sensor_fwhm_nm: Optional[Tensor] = None,
        rt_overrides: Optional[Dict[str, float | Tensor]] = None,
        return_intermediates: bool = False,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """Approximate inverse mapping back to high-resolution reflectance."""

        if sensor_measurement.ndim != 2:
            raise ValueError("sensor_measurement must be (batch, bands)")
        device = sensor_measurement.device
        dtype = sensor_measurement.dtype
        centers, fwhm, matrix, pinv = self._resolve_sensor(
            sensor_centers_nm, sensor_fwhm_nm, dtype, device
        )
        hi_grid = self.hi_wvl_nm.to(device=device, dtype=dtype)
        params = dict(self.rt_params)
        if rt_overrides:
            params.update(rt_overrides)
        intermediates: Dict[str, Tensor] = {
            "sensor_matrix": matrix,
            "sensor_centers_nm": centers,
            "sensor_fwhm_nm": fwhm,
        }

        hi_signal = sensor_measurement @ pinv.transpose(-1, -2)
        if self.mode == "swir_reflectance":
            reflectance = hi_signal.clamp(0.0, 1.0)
        elif self.mode == "swir_reflectance_to_radiance":
            cfg = SWIRRTConfig(
                tau=float(params.get("tau", 0.9)),
                scale=float(params.get("scale", 1.0)),
                noise_std=float(params.get("noise_std", 0.0)),
            )
            solar = params.get("solar_irradiance")
            if solar is None:
                solar = self._solar_irradiance(device, dtype)
            else:
                solar = torch.as_tensor(solar, dtype=dtype, device=device)
            reflectance = swir_radiance_to_reflectance(hi_signal, solar, cfg.tau, cfg.scale)
            intermediates["radiance_hi"] = hi_signal
            intermediates["solar_irradiance"] = solar
        else:
            planck = params.get("planck_L")
            if planck is None:
                planck = torch.exp(-((hi_grid - hi_grid.mean()) ** 2) / (2 * 250.0**2))
            else:
                planck = torch.as_tensor(planck, dtype=dtype, device=device)
            gain = float(params.get("gain", 1e-3))
            offset = float(params.get("offset", 1.0))
            temp_est = ((hi_signal / (planck.clamp_min(1e-6))).mean(dim=-1) - offset) / max(gain, 1e-6)
            emissivity = params.get("emissivity")
            if emissivity is None:
                emissivity = torch.ones_like(hi_signal)
            else:
                emissivity = torch.as_tensor(emissivity, dtype=dtype, device=device)
            reflectance = (1.0 - emissivity).clamp(0.0, 1.0)
            intermediates["radiance_hi"] = hi_signal
            intermediates["temperature"] = temp_est.unsqueeze(-1)
        intermediates["reflectance_hi"] = reflectance
        return reflectance, (intermediates if return_intermediates else None)
