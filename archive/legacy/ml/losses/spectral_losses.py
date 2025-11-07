from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F


# ----------------------------- SAM -----------------------------

def sam_loss(
    f_hat: Tensor,      # [B,K]
    f_true: Tensor,     # [B,K]
    eps: float = 1e-8,
    use_angle: bool = False,
    reduction: str = "mean",
) -> Tensor:
    """
    Spectral Angle Mapper loss.

    If use_angle=False: 1 - cosine similarity (bounded [0,2])
    If use_angle=True:  angle in radians (arccos of cosine)
    """
    x = f_hat
    y = f_true
    x_norm = x.norm(dim=-1).clamp_min(eps)
    y_norm = y.norm(dim=-1).clamp_min(eps)
    cos = (x * y).sum(dim=-1) / (x_norm * y_norm)
    cos = cos.clamp(-1.0, 1.0)
    if use_angle:
        loss = torch.arccos(cos)
    else:
        loss = 1.0 - cos
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


# ----------------------------- Curvature / TV -----------------------------

@dataclass
class CurvatureConfig:
    lambda_weights: Optional[Tensor] = None  # [K] weights; smaller inside known absorption windows
    reduction: str = "mean"


def curvature_loss(
    f_hat: Tensor,                # [B,K]
    cfg: CurvatureConfig = CurvatureConfig(),
) -> Tensor:
    """
    Discrete second-difference energy: sum_k (f[k+1] - 2 f[k] + f[k-1])^2
    """
    B, K = f_hat.shape
    d2 = f_hat[:, 2:] - 2.0 * f_hat[:, 1:-1] + f_hat[:, :-2]  # [B,K-2]
    e = d2 ** 2
    if cfg.lambda_weights is not None:
        w = cfg.lambda_weights.to(f_hat.device, f_hat.dtype)
        assert w.shape == (K,), "lambda_weights must be shape [K]"
        w2 = w[1:-1]
        e = e * w2
    if cfg.reduction == "mean":
        return e.mean()
    if cfg.reduction == "sum":
        return e.sum()
    return e


# ----------------------------- Box constraints -----------------------------

@dataclass
class BoxPenaltyConfig:
    lo: float = 0.0
    hi: float = 1.0
    alpha_plus: float = 0.5     # extra weight on >hi violations
    reduction: str = "mean"


def box_penalty(
    f_hat: Tensor,
    cfg: BoxPenaltyConfig = BoxPenaltyConfig(),
) -> Tensor:
    """
    Quadratic penalties outside [lo, hi], with asymmetric weight on upper violations.
    """
    below = F.relu(cfg.lo - f_hat)
    above = F.relu(f_hat - cfg.hi)
    loss = (below ** 2) + cfg.alpha_plus * (above ** 2)
    if cfg.reduction == "mean":
        return loss.mean()
    if cfg.reduction == "sum":
        return loss.sum()
    return loss


def violation_rate(
    f_hat: Tensor,
    lo: float = 0.0,
    hi: float = 1.0,
) -> Tensor:
    """Percentage of bins outside [lo,hi]."""
    B, K = f_hat.shape
    v = ((f_hat < lo) | (f_hat > hi)).to(f_hat.dtype).mean()
    return v


# ----------------------------- CR loss (differentiable) -----------------------------

def _savgol_kernel(window_length: int, polyorder: int, device, dtype) -> Tensor:
    """
    Returns a [1,1,W] 1D convolution kernel for Savitzky–Golay smoothing.

    We compute the pseudo-inverse solution for polynomial LS fit within the window,
    then take the coefficients corresponding to the center sample (smoothing filter).
    """
    import numpy as np
    if window_length % 2 != 1:
        raise ValueError("window_length must be odd")
    if polyorder >= window_length:
        raise ValueError("polyorder must be < window_length")

    half = window_length // 2
    x = np.arange(-half, half + 1, dtype=np.float64)  # centered window
    # Vandermonde
    A = np.vstack([x ** i for i in range(polyorder + 1)]).T  # [W, P+1]
    pinv = np.linalg.pinv(A)                                 # [P+1, W]
    coeffs = pinv[0]                                         # smoothing at center -> first row
    k = torch.tensor(coeffs, device=device, dtype=dtype).view(1, 1, -1)  # [1,1,W]
    return k


def _smooth_continuum_above(f: Tensor, window: int = 31, poly: int = 3, beta: float = 10.0) -> Tensor:
    """
    Compute a smooth continuum c >= f by (i) Savitzky–Golay smoothing and
    (ii) raising it to at least f using softplus with temperature 1/beta:

        c = s + softplus(f - s, beta)

    where s = SG(f). This enforces c >= f while remaining differentiable.
    """
    B, K = f.shape
    pad = window // 2
    k = _savgol_kernel(window, poly, device=f.device, dtype=f.dtype)  # [1,1,W]
    x = f.unsqueeze(1)  # [B,1,K]
    s = torch.conv1d(F.pad(x, (pad, pad), mode="reflect"), k)  # [B,1,K]
    s = s.squeeze(1)
    # softplus lifting
    c = s + F.softplus(f - s, beta=beta)
    return c


def cr_loss_smooth(
    f_hat: Tensor,           # [B,K]
    f_true: Tensor,          # [B,K]
    window: int = 31,
    poly: int = 3,
    beta: float = 10.0,
    clamp: Tuple[float, float] = (0.0, 1.0),
    reduction: str = "mean",
) -> Tensor:
    """
    Differentiable CR loss using a smooth upper envelope as continuum.
    """
    c_hat = _smooth_continuum_above(f_hat, window, poly, beta)           # [B,K]
    c_true = _smooth_continuum_above(f_true, window, poly, beta)
    cr_h = (f_hat / c_hat.clamp_min(1e-6)).clamp(*clamp)
    cr_t = (f_true / c_true.clamp_min(1e-6)).clamp(*clamp)
    loss = (cr_h - cr_t) ** 2
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
