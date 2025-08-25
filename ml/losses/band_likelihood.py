from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import Tensor
import torch.nn.functional as F


Reduction = Literal["none", "mean", "sum"]


@dataclass
class HeteroHuberConfig:
    """Config for heteroscedastic Huber negative log-likelihood."""
    kappa: float = 1.0                 # transition between L2 and L1 in standardized residual units
    eps: float = 1e-8                  # numerical stability
    reduction: Reduction = "mean"
    include_log_sigma: bool = True     # add log(sigma) term to prevent trivial sigma->0


def hetero_huber_nll(
    residual: Tensor,        # shape [..., N]
    sigma: Tensor,           # shape broadcastable to residual
    cfg: HeteroHuberConfig = HeteroHuberConfig(),
    mask: Optional[Tensor] = None,     # boolean mask same shape (True=valid)
) -> Tensor:
    """
    Negative "log-likelihood" style Huber with per-sample sigma. Standardizes residuals by sigma,
    uses Huber penalty on standardized residual, and (optionally) adds log(sigma).

    Returns: scalar if reduction!='none', else per-element tensor.
    """
    if mask is not None:
        residual = torch.where(mask, residual, torch.zeros_like(residual))
        sigma = torch.where(mask, sigma, torch.ones_like(sigma))

    sigma = sigma.clamp_min(cfg.eps)
    r = residual / sigma
    abs_r = r.abs()
    quad = 0.5 * (r ** 2)
    lin = cfg.kappa * (abs_r - 0.5 * cfg.kappa)
    loss = torch.where(abs_r <= cfg.kappa, quad, lin)
    if cfg.include_log_sigma:
        loss = loss + torch.log(sigma)

    if mask is not None:
        denom = mask.to(loss.dtype).sum().clamp_min(1.0) if cfg.reduction == "mean" else 1.0
    else:
        denom = 1.0

    if cfg.reduction == "mean":
        return loss.sum() / denom
    if cfg.reduction == "sum":
        return loss.sum()
    return loss


@dataclass
class StudentTConfig:
    """Student-t negative log-likelihood with per-sample scale sigma and dof nu."""
    nu: float = 5.0                    # degrees of freedom (>0); smaller = heavier tails
    eps: float = 1e-8
    reduction: Reduction = "mean"


def student_t_nll(
    residual: Tensor,        # shape [..., N]
    sigma: Tensor,           # broadcastable
    cfg: StudentTConfig = StudentTConfig(),
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    NLL for Student-t with location=0, scale=sigma, dof=nu.

    NLL(x) = 0.5*log(nu*pi) + log(sigma) + 0.5*(nu+1)*log(1 + (x/sigma)^2 / nu)
    """
    if cfg.nu <= 0:
        raise ValueError("StudentTConfig.nu must be > 0")
    if mask is not None:
        residual = torch.where(mask, residual, torch.zeros_like(residual))
        sigma = torch.where(mask, sigma, torch.ones_like(sigma))

    sigma = sigma.clamp_min(cfg.eps)
    r = residual / sigma
    nll = 0.5 * torch.log(torch.tensor(cfg.nu * torch.pi, device=residual.device, dtype=residual.dtype))
    nll = nll + torch.log(sigma)
    nll = nll + 0.5 * (cfg.nu + 1.0) * torch.log1p((r ** 2) / cfg.nu)

    if mask is not None:
        denom = mask.to(nll.dtype).sum().clamp_min(1.0) if cfg.reduction == "mean" else 1.0
    else:
