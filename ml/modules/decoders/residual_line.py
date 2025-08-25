from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
from torch import nn, Tensor

from ....ml.interfaces import DecoderProtocol


# --------------------- Dictionary Builders ---------------------

def _gaussian_dict(K: int, centers01: Tensor, sigma01: float) -> Tensor:
    """
    Build a Gaussian atom dictionary over K wavelengths normalized to [0,1].
    Each atom j is a Gaussian centered at centers01[j] with std sigma01.
    Returns D: [K, M]
    """
    u = torch.linspace(0.0, 1.0, K, device=centers01.device, dtype=centers01.dtype).unsqueeze(-1)  # [K,1]
    c = centers01.view(1, -1)  # [1, M]
    D = torch.exp(-0.5 * ((u - c) / max(sigma01, 1e-6)) ** 2)  # [K, M]
    # Normalize atoms to unit L2 (helps with coefficient scaling)
    D = D / (D.norm(dim=0, keepdim=True).clamp_min(1e-8))
    return D


def _cosine_dict(K: int, num_freqs: int) -> Tensor:
    """
    Cosine dictionary with increasing frequencies across [0,1].
    Returns D: [K, M], M = num_freqs
    """
    u = torch.linspace(0.0, 1.0, K).unsqueeze(-1)  # [K,1]
    freqs = torch.arange(1, num_freqs + 1, dtype=u.dtype).view(1, -1)  # [1,M]
    D = torch.cos(2 * math.pi * u * freqs)  # [K, M]
    D = D / (D.norm(dim=0, keepdim=True).clamp_min(1e-8))
    return D


# --------------------- Config ---------------------

@dataclass
class ResidualLineHeadConfig:
    """
    Sparse residual 'line' head: r(λ) = D a, where D is a dictionary of narrow atoms.

    Args:
        K:                number of wavelength bins
        kind:             'gaussian' | 'cosine'
        num_atoms:        number of atoms (M)
        gaussian_sigma_nm: approximate std in nm for Gaussian atoms (used if kind='gaussian')
        lambda_min_nm/max_nm: to map sigma_nm -> sigma01
        windows_mask:     optional [K] boolean mask of where residuals can be placed (True=allowed)
        d_in:             latent dimension input for coefficient head
        hidden:           hidden size in the small MLP
        layers:           number of layers in coefficient head
        output_scale:     final scaling on residual before addition (stability)
    """
    K: int
    kind: str = "gaussian"
    num_atoms: int = 128
    gaussian_sigma_nm: float = 5.0
    lambda_min_nm: float = 400.0
    lambda_max_nm: float = 2500.0
    windows_mask: Optional[Tensor] = None
    d_in: int = 512
    hidden: int = 256
    layers: int = 2
    output_scale: float = 1.0


# --------------------- Residual Head ---------------------

class _CoeffHead(nn.Module):
    def __init__(self, d_in: int, M: int, hidden: int, layers: int):
        super().__init__()
        if layers <= 1:
            self.net = nn.Linear(d_in, M)
        else:
            blocks = [nn.Linear(d_in, hidden), nn.GELU()]
            for _ in range(layers - 2):
                blocks += [nn.Linear(hidden, hidden), nn.GELU()]
            blocks += [nn.Linear(hidden, M)]
            self.net = nn.Sequential(*blocks)

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class ResidualLineHead(nn.Module):
    """
    Compute sparse residual r = D a from latent z, where D is a fixed dictionary.

    Exposes:
        - forward(z) -> r [B,K]
        - last_coeffs: coefficients a [B,M] from the last forward (for L1 penalties in losses)
    """
    def __init__(self, cfg: ResidualLineHeadConfig):
        super().__init__()
        self.cfg = cfg
        self.last_coeffs: Optional[Tensor] = None

        # Build dictionary
        if cfg.kind.lower() == "gaussian":
            centers01 = torch.linspace(0.0, 1.0, cfg.num_atoms)
            sigma01 = cfg.gaussian_sigma_nm / max(1e-6, (cfg.lambda_max_nm - cfg.lambda_min_nm))
            D = _gaussian_dict(cfg.K, centers01, sigma01)
        elif cfg.kind.lower() == "cosine":
            D = _cosine_dict(cfg.K, cfg.num_atoms)
        else:
            raise ValueError(f"Unsupported dictionary kind: {cfg.kind}")

        # Optional windowing (zero out atoms outside allowed windows)
        if cfg.windows_mask is not None:
            if cfg.windows_mask.shape != (cfg.K,):
                raise ValueError("windows_mask must be shape [K]")
            mask = cfg.windows_mask.to(dtype=D.dtype)
            D = D * mask.unsqueeze(-1)

        # Re-orthonormalize columns to avoid degenerate scaling after windowing
        # Gram-Schmidt (simple, stable enough for modest M)
        with torch.no_grad():
            for j in range(D.shape[1]):
                v = D[:, j]
                for i in range(j):
                    u = D[:, i]
                    v = v - (v @ u) * u
                v = v / v.norm().clamp_min(1e-8)
                D[:, j] = v

        self.register_buffer("D", D, persistent=False)  # [K,M]
        self.head = _CoeffHead(cfg.d_in, D.shape[1], cfg.hidden, cfg.layers)

    def forward(self, z: Tensor) -> Tensor:
        a = self.head(z)                 # [B,M]
        self.last_coeffs = a
        r = a @ self.D.T                 # [B,K]
        return r * self.cfg.output_scale


# --------------------- Wrapper to augment a base decoder ---------------------

class ResidualAugmentedDecoder(nn.Module, DecoderProtocol):
    """
    Wrap any base DecoderProtocol and add a residual 'line' head to its output:

        f_total = f_base + r(z)

    The residual is intended for very narrow absorptions and should be regularized
    with an L1 penalty on `head.last_coeffs` in the training loss.
    """
    def __init__(self, base: DecoderProtocol, head: ResidualLineHead):
        super().__init__()
        # register as modules to save parameters properly
        self.base = base  # type: ignore[arg-type]
        self.head = head

    def forward(self, z: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        f_base, c_hat = self.base(z)          # [B,K], [B,K]|None
        r = self.head(z)                      # [B,K]
        f_hat = f_base + r
        # Clamp lightly to avoid runaway values while preserving negative allowance for loss penalties
        f_hat = f_hat.clamp(-0.1, 1.1)
        return f_hat, c_hat
