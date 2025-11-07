# hsi_fm/physics/renderer.py
"""Simple, differentiable physics renderer implemented with PyTorch.

Features
--------
- SWIR-style mock radiative transfer: L_hi = τ * E0 * R_hi + L_path
- SRF-aware band projection using normalized SRF matrix (K, L)
- Ridge-regularized pseudo-inverse for stable band → hi-grid inversion
- CPU/GPU safe, dtype/device propagation
- Backward-compatible list-based APIs (`render`, `invert`) plus
  tensor-first APIs (`render_torch`, `invert_torch`)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union, Tuple

import torch
from torch import Tensor

from hsi_fm.physics.srf import (
    SpectralResponseFunction,
    gaussian_srf_torch,
    tabulated_srf_torch,
    convolve_srf_torch,
    normalise_srf,           # used if callers pass prebuilt matrices
)

Number = float
VectorLike = Sequence[Number]
MatrixLike = Sequence[VectorLike]


# ---------------------------------------------------------------------------
# Utility: list↔tensor interop
# ---------------------------------------------------------------------------

def _ensure_1d_tensor(x: Union[Tensor, Sequence[Number]], *, dtype: torch.dtype, device: torch.device) -> Tensor:
    if isinstance(x, Tensor):
        t = x
        if t.ndim != 1:
            raise ValueError("Expected 1D tensor for wavelength grid.")
        return t.to(dtype=dtype, device=device)
    t = torch.tensor(x, dtype=dtype, device=device)
    if t.ndim != 1:
        raise ValueError("Expected 1D sequence for wavelength grid.")
    return t


def _ensure_2d_tensor(x: Union[Tensor, VectorLike, MatrixLike], *, dtype: torch.dtype, device: torch.device) -> Tensor:
    if isinstance(x, Tensor):
        t = x
        if t.ndim == 1:
            t = t.unsqueeze(0)
        if t.ndim != 2:
            raise ValueError("Expected (B, L) or (L,) for spectra.")
        return t.to(dtype=dtype, device=device)
    # list/tuple path
    arr = list(x)  # type: ignore[arg-type]
    if len(arr) == 0:
        return torch.empty(0, 0, dtype=dtype, device=device)
    if isinstance(arr[0], (list, tuple)):
        t = torch.tensor(arr, dtype=dtype, device=device)
        if t.ndim != 2:
            raise ValueError("Matrix-like spectra must be 2D.")
        return t
    # 1D → 2D
    t = torch.tensor(arr, dtype=dtype, device=device).unsqueeze(0)
    return t


def _tolist_2d(x: Tensor) -> List[List[Number]]:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    return x.detach().cpu().tolist()


# ---------------------------------------------------------------------------
# Atmospheric state
# ---------------------------------------------------------------------------

@dataclass
class AtmosphericState:
    """Mock radiative-transfer state for SWIR-like paths."""
    irradiance: Number = 1.0      # E0
    transmittance: Number = 1.0   # τ
    path_radiance: Number = 0.0   # L_path

    def as_scalars(self) -> Tuple[float, float, float]:
        return float(self.irradiance), float(self.transmittance), float(self.path_radiance)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class LabToSensorRenderer:
    """
    Render lab high-res reflectance to sensor bandspace using SRFs and a mock RT model.
    Also provides a differentiable inverse via ridge-regularized pseudo-inverse.

    Usage (tensor):
        renderer = LabToSensorRenderer(srf)   # SRF defined on provided hi grid later
        L_sensor = renderer.render_torch(R_hi, wavelengths_nm, atm)
        R_hat    = renderer.invert_torch(L_sensor, wavelengths_nm, atm, ridge=1e-3)

    Usage (lists, backward-compatible):
        L_sensor_list = renderer.render(R_list, wavelengths_list, atm)
        R_hat_list    = renderer.invert(L_list, wavelengths_list, atm)
    """

    def __init__(self, srf: SpectralResponseFunction | Tensor):
        """
        Parameters
        ----------
        srf : SpectralResponseFunction or Tensor
            - If SpectralResponseFunction: SRFs will be built/normalized on the
              provided hi wavelength grid at call-time.
            - If Tensor: prebuilt SRF matrix (K, L) assumed to be aligned to
              any wavelengths you pass later (you can still pass wavelengths for checks).
        """
        self._srf_def = srf  # keep as-is; build matrix at call time if needed

    # ------------------------------- Core math --------------------------------

    @staticmethod
    def _radiance_forward(R_hi: Tensor, E0: float, tau: float, Lp: float) -> Tensor:
        # L = τ * E0 * R + L_path
        return R_hi.mul(tau * E0).add(Lp)

    @staticmethod
    def _reflectance_inverse(L_hi: Tensor, E0: float, tau: float, Lp: float) -> Tensor:
        # R = (L - L_path) / (τ * E0)
        denom = max(tau * E0, 1e-12)
        return (L_hi - Lp) / denom

    @staticmethod
    def _build_srf_matrix(
        srf_def: SpectralResponseFunction | Tensor,
        wavelengths_nm: Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """
        Returns SRF matrix (K, L) normalized on the provided wavelength grid.
        """
        if isinstance(srf_def, Tensor):
            # Assume caller ensures alignment; still cast/normalize defensively.
            M = srf_def.to(dtype=dtype, device=device)
            # If it's already normalized (rows sum to 1), this is idempotent.
            M = normalise_srf(M, wavelengths_nm=wavelengths_nm)
            return M

        # SpectralResponseFunction path: build Gaussian or normalize tabulated
        if srf_def.responses is None:
            centers = torch.tensor(srf_def.centers, dtype=dtype, device=device)
            widths  = torch.tensor(srf_def.widths,  dtype=dtype, device=device)
            M = gaussian_srf_torch(centers, widths, wavelengths_nm, normalize=True)  # (K, L)
        else:
            M = tabulated_srf_torch(
                srf_def.responses.to(dtype=dtype, device=device), wavelengths_nm, normalize=True
            )
        return M

    @staticmethod
    def _ridge_pseudoinverse(M: Tensor, ridge: float = 1e-3) -> Tensor:
        """
        Compute (M^T M + λI)^(-1) M^T without forming an explicit inverse:
        Solve (M^T M + λI) X = M^T  →  X = (L,K)
        """
        K, L = M.shape
        Mt = M.transpose(0, 1)               # (L, K)
        gram = Mt @ M                         # (L, L)
        if ridge > 0:
            gram = gram + ridge * torch.eye(L, dtype=M.dtype, device=M.device)
        # Solve for X in (L,L) @ X = (L,K)  → X is (L,K)
        X = torch.linalg.solve(gram, Mt)
        return X  # (L, K)

    # ----------------------------- Tensor API ---------------------------------

    def render_torch(
        self,
        reflectance_hi: Tensor,
        wavelengths_nm: Tensor,
        atmosphere: Optional[AtmosphericState] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        reflectance_hi : (B, L) or (L,)
        wavelengths_nm : (L,)
        atmosphere     : AtmosphericState

        Returns
        -------
        L_sensor : (B, K)
        """
        atmosphere = atmosphere or AtmosphericState()
        # dtype/device resolution from input spectra or wavelengths
        device = reflectance_hi.device if isinstance(reflectance_hi, Tensor) else torch.device("cpu")
        dtype = reflectance_hi.dtype if isinstance(reflectance_hi, Tensor) else torch.float32

        R_hi = _ensure_2d_tensor(reflectance_hi, dtype=dtype, device=device)  # (B, L)
        wvl = _ensure_1d_tensor(wavelengths_nm, dtype=R_hi.dtype, device=R_hi.device)

        # Build/normalize SRF matrix on this grid
        M = self._build_srf_matrix(self._srf_def, wvl, dtype=R_hi.dtype, device=R_hi.device)  # (K, L)

        # Forward RT on hi grid, then convolve to sensor bands
        E0, tau, Lp = atmosphere.as_scalars()
        L_hi = self._radiance_forward(R_hi, E0, tau, Lp)              # (B, L)
        L_sensor = convolve_srf_torch(L_hi, wvl, M)                   # (B, K)
        return L_sensor

    def invert_torch(
        self,
        radiance_sensor: Tensor,
        wavelengths_nm: Tensor,
        atmosphere: Optional[AtmosphericState] = None,
        ridge: float = 1e-3,
    ) -> Tensor:
        """
        Parameters
        ----------
        radiance_sensor : (B, K) or (K,)
        wavelengths_nm  : (L,)
        atmosphere      : AtmosphericState
        ridge           : ridge term for stable inversion (meters SRF ill-conditioning)

        Returns
        -------
        reflectance_hi_hat : (B, L)
        """
        atmosphere = atmosphere or AtmosphericState()
        # Choose dtype/device from input
        device = radiance_sensor.device if isinstance(radiance_sensor, Tensor) else torch.device("cpu")
        dtype = radiance_sensor.dtype if isinstance(radiance_sensor, Tensor) else torch.float32

        L_sensor = _ensure_2d_tensor(radiance_sensor, dtype=dtype, device=device)  # (B, K)
        wvl = _ensure_1d_tensor(wavelengths_nm, dtype=L_sensor.dtype, device=L_sensor.device)

        # SRF matrix
        M = self._build_srf_matrix(self._srf_def, wvl, dtype=L_sensor.dtype, device=L_sensor.device)  # (K, L)
        # Compute ridge pseudo-inverse (L, K), project back to hi-grid radiance
        X = self._ridge_pseudoinverse(M, ridge=ridge)                         # (L, K)
        L_hi_hat = L_sensor @ X.transpose(0, 1)                               # (B, L)

        E0, tau, Lp = atmosphere.as_scalars()
        R_hi_hat = self._reflectance_inverse(L_hi_hat, E0, tau, Lp)           # (B, L)
        return R_hi_hat

    # ------------------------- Back-compat list API ---------------------------

    def render(
        self,
        reflectance: Sequence[Sequence[Number] | Number],
        wavelengths: Sequence[Number],
        atmosphere: Optional[AtmosphericState] = None,
    ) -> List[List[Number]]:
        """List-friendly wrapper around `render_torch`."""
        # Resolve dtype/device from CPU tensors
        R_hi = _ensure_2d_tensor(reflectance, dtype=torch.float32, device=torch.device("cpu"))
        wvl = _ensure_1d_tensor(wavelengths, dtype=R_hi.dtype, device=R_hi.device)
        L_sensor = self.render_torch(R_hi, wvl, atmosphere)
        return _tolist_2d(L_sensor)

    def invert(
        self,
        radiance: Sequence[Sequence[Number] | Number],
        wavelengths: Sequence[Number],
        atmosphere: Optional[AtmosphericState] = None,
        ridge: float = 1e-3,
    ) -> List[List[Number]]:
        """List-friendly wrapper around `invert_torch`."""
        L = _ensure_2d_tensor(radiance, dtype=torch.float32, device=torch.device("cpu"))
        wvl = _ensure_1d_tensor(wavelengths, dtype=L.dtype, device=L.device)
        R_hat = self.invert_torch(L, wvl, atmosphere, ridge=ridge)
        return _tolist_2d(R_hat)


__all__ = ["LabToSensorRenderer", "AtmosphericState"]
