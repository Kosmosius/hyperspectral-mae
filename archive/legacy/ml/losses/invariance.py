from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F


# ----------------- Gradient Reversal -----------------

class _GR(Function := torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, lamb: float):
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return -ctx.lamb * grad_output, None


class GRL(nn.Module):
    """Gradient Reversal Layer with dynamic lambda."""
    def __init__(self, lamb: float = 1.0):
        super().__init__()
        self.lamb = lamb

    def forward(self, x: Tensor) -> Tensor:
        return _GR.apply(x, float(self.lamb))


# ----------------- Sensor Adversary -----------------

@dataclass
class AdversaryConfig:
    in_dim: int
    num_domains: int
    hidden: int = 256
    layers: int = 2
    dropout: float = 0.1
    grl_lambda: float = 1.0


class SensorAdversary(nn.Module):
    """
    Domain adversary: GRL -> small MLP -> logits over sensor/domain IDs.
    """
    def __init__(self, cfg: AdversaryConfig):
        super().__init__()
        self.grl = GRL(cfg.grl_lambda)
        if cfg.layers <= 1:
            self.clf = nn.Linear(cfg.in_dim, cfg.num_domains)
        else:
            blocks = [nn.Linear(cfg.in_dim, cfg.hidden), nn.GELU(), nn.Dropout(cfg.dropout)]
            for _ in range(cfg.layers - 2):
                blocks += [nn.Linear(cfg.hidden, cfg.hidden), nn.GELU(), nn.Dropout(cfg.dropout)]
            blocks += [nn.Linear(cfg.hidden, cfg.num_domains)]
            self.clf = nn.Sequential(*blocks)

    def forward(self, z: Tensor) -> Tensor:
        return self.clf(self.grl(z))  # [B,num_domains]


# ----------------- HSIC / MMD -----------------

def _rbf_kernel(x: Tensor, y: Optional[Tensor] = None, gamma: Optional[float] = None) -> Tensor:
    """
    RBF kernel matrix. If y is None -> self-kernel.
    gamma is 1/(2*sigma^2). If None, uses median heuristic on x (and y if provided).
    """
    if y is None:
        y = x
    if gamma is None:
        with torch.no_grad():
            xx = (x[:, None, :] - y[None, :, :]) ** 2
            d2 = xx.sum(dim=-1)
            med = torch.median(d2[d2 > 0]).clamp_min(1e-12)
            gamma = 1.0 / (2.0 * med)
    else:
        gamma = float(gamma)
    d2 = torch.cdist(x, y) ** 2
    return torch.exp(-gamma * d2)


def hsic_rbf(x: Tensor, y: Tensor, gamma_x: Optional[float] = None, gamma_y: Optional[float] = None) -> Tensor:
    """
    Unbiased HSIC estimator with RBF kernels.
    Returns a scalar; higher => more dependence.
    """
    n = x.shape[0]
    if n < 4:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    K = _rbf_kernel(x, gamma=gamma_x)
    L = _rbf_kernel(y, gamma=gamma_y)
    H = torch.eye(n, device=x.device, dtype=x.dtype) - (1.0 / n) * torch.ones((n, n), device=x.device, dtype=x.dtype)
    KH = K @ H
    LH = L @ H
    hsic = (KH * LH).sum() / ((n - 1) ** 2)
    return hsic


def mmd_rbf(x: Tensor, y: Tensor, gamma: Optional[float] = None) -> Tensor:
    """
    Squared MMD with RBF kernel (unbiased estimator). Returns scalar.
    """
    Kxx = _rbf_kernel(x, x, gamma)
    Kyy = _rbf_kernel(y, y, gamma)
    Kxy = _rbf_kernel(x, y, gamma)
    n = x.shape[0]
    m = y.shape[0]
    # Remove diagonals for unbiased estimate
    sum_xx = (Kxx.sum() - Kxx.diag().sum()) / (n * (n - 1))
    sum_yy = (Kyy.sum() - Kyy.diag().sum()) / (m * (m - 1))
    sum_xy = Kxy.mean()
    return sum_xx + sum_yy - 2.0 * sum_xy
