from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
from torch import nn, Tensor

from ....ml.interfaces import DecoderProtocol


# --------------------- Branch Utilities ---------------------

class Sine(nn.Module):
    def __init__(self, w0: float = 30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.w0 * x)


def _init_siren(m: nn.Module, c: float = 6.0, w0: float = 30.0):
    if isinstance(m, nn.Linear):
        fan_in = m.weight.size(1)
        bound = math.sqrt(c / fan_in) / w0
        with torch.no_grad():
            m.weight.uniform_(-bound, bound)
            if m.bias is not None:
                m.bias.fill_(0.0)


# --------------------- Config ---------------------

@dataclass
class DeepONetDecoderConfig:
    """
    DeepONet-style decoder f(λ) = < trunk(z), branch(λ) >.

    Args:
        d_in:           latent dim (z)
        K:              number of canonical wavelength bins
        lambdas01:      [K] tensor (0..1); if None, uses linspace(0,1,K)
        trunk_dim:      feature size D_t
        trunk_layers:   layers for trunk MLP (>=1)
        trunk_hidden:   hidden size for trunk MLP
        branch:         'rff' | 'siren'
        rff_dim:        number of RFF frequencies (each produces sin+cos)
        rff_fmin/fmax:  min/max frequency for RFF (cycles over [0,1])
        siren_w0:       first-layer frequency scale for SIREN
        siren_depth:    depth of SIREN branch net
        siren_width:    width of SIREN branch net
        activation:     output activation for reflectance (identity|sigmoid|softplus_clamp)
        clamp_min/max:  final clamp range
        use_continuum:  also predict continuum c_hat[K]
        continuum_degree: polynomial degree if no custom basis
        continuum_basis: optional custom [K,M_c]
        continuum_range: allowed range for c_hat
    """
    d_in: int
    K: int
    lambdas01: Optional[Tensor] = None
    trunk_dim: int = 128
    trunk_layers: int = 2
    trunk_hidden: int = 256
    branch: str = "rff"                   # 'rff' or 'siren'
    rff_dim: int = 64
    rff_fmin: float = 2.0
    rff_fmax: float = 64.0
    siren_w0: float = 20.0
    siren_depth: int = 3
    siren_width: int = 128
    activation: str = "identity"
    clamp_min: float = -0.05
    clamp_max: float = 1.05
    use_continuum: bool = True
    continuum_degree: int = 5
    continuum_basis: Optional[Tensor] = None
    continuum_range: Tuple[float, float] = (0.7, 1.3)


# --------------------- DeepONet Decoder ---------------------

class _Trunk(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int, layers: int):
        super().__init__()
        if layers <= 1:
            self.net = nn.Linear(d_in, d_out)
        else:
            blocks = [nn.Linear(d_in, hidden), nn.GELU()]
            for _ in range(layers - 2):
                blocks += [nn.Linear(hidden, hidden), nn.GELU()]
            blocks += [nn.Linear(hidden, d_out)]
            self.net = nn.Sequential(*blocks)

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)  # [B, d_out]


class _BranchRFF(nn.Module):
    def __init__(self, K: int, D_out: int, rff_dim: int, fmin: float, fmax: float):
        super().__init__()
        # Static sin/cos features for fixed λ grid
        self.D_out = D_out
        self.rff_dim = rff_dim
        self.register_buffer("freqs", torch.logspace(math.log10(fmin), math.log10(fmax), rff_dim), persistent=False)
        # learned linear projection to D_out
        self.proj = nn.Linear(2 * rff_dim, D_out)

    def forward(self, lambdas01: Tensor) -> Tensor:
        # lambdas01: [K]
        arg = 2 * math.pi * lambdas01.unsqueeze(-1) * self.freqs  # [K, M]
        feats = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)  # [K, 2M]
        return self.proj(feats)  # [K, D_out]


class _BranchSIREN(nn.Module):
    def __init__(self, D_out: int, depth: int, width: int, w0: float):
        super().__init__()
        layers = []
        layers += [nn.Linear(1, width), Sine(w0)]
        for _ in range(depth - 2):
            layers += [nn.Linear(width, width), Sine(w0)]
        layers += [nn.Linear(width, D_out)]
        self.net = nn.Sequential(*layers)
        self.apply(lambda m: _init_siren(m, w0=w0))

    def forward(self, lambdas01: Tensor) -> Tensor:
        # lambdas01: [K] -> [K,1]
        x = lambdas01.view(-1, 1)
        return self.net(x)  # [K, D_out]


def _make_poly_basis01(K: int, degree: int, device, dtype) -> Tensor:
    u = torch.linspace(0.0, 1.0, K, device=device, dtype=dtype)
    cols = [torch.ones_like(u)]
    for d in range(1, degree + 1):
        cols.append(u ** d)
    return torch.stack(cols, dim=-1)  # [K, degree+1]


class DeepONetDecoder(nn.Module, DecoderProtocol):
    """
    DeepONet decoder with selectable branch (RFF or SIREN) and optional continuum head.
    """
    def __init__(self, cfg: DeepONetDecoderConfig):
        super().__init__()
        self.cfg = cfg

        self.K = cfg.K
        if cfg.lambdas01 is None:
            lambdas01 = torch.linspace(0.0, 1.0, cfg.K)
        else:
            lambdas01 = cfg.lambdas01
            assert lambdas01.shape == (cfg.K,), "lambdas01 must be shape [K]"
        self.register_buffer("lambdas01", lambdas01, persistent=False)

        # Trunk
        self.trunk = _Trunk(cfg.d_in, cfg.trunk_dim, cfg.trunk_hidden, cfg.trunk_layers)

        # Branch
        btype = cfg.branch.lower()
        if btype == "rff":
            self.branch = _BranchRFF(cfg.K, cfg.trunk_dim, cfg.rff_dim, cfg.rff_fmin, cfg.rff_fmax)
        elif btype == "siren":
            self.branch = _BranchSIREN(cfg.trunk_dim, cfg.siren_depth, cfg.siren_width, cfg.siren_w0)
        else:
            raise ValueError(f"Unsupported branch type: {cfg.branch}")

        # Continuum
        self.use_continuum = cfg.use_continuum
        if self.use_continuum:
            if cfg.continuum_basis is not None:
                Cb = cfg.continuum_basis
                assert Cb.shape[0] == cfg.K
            else:
                Cb = _make_poly_basis01(cfg.K, cfg.continuum_degree, device=torch.device("cpu"), dtype=torch.get_default_dtype())
            self.register_buffer("cont_basis", Cb, persistent=False)
            self.cont_head = nn.Sequential(
                nn.Linear(cfg.d_in, max(64, cfg.trunk_hidden // 2)),
                nn.GELU(),
                nn.Linear(max(64, cfg.trunk_hidden // 2), Cb.shape[1]),
            )

        # Output activation
        act = cfg.activation.lower()
        if act not in ("identity", "sigmoid", "softplus_clamp"):
            raise ValueError(f"Unsupported activation: {cfg.activation}")
        self._act_name = act
        self.softplus = nn.Softplus(beta=1.0)

    def _apply_activation(self, x: Tensor) -> Tensor:
        if self._act_name == "identity":
            return x.clamp(self.cfg.clamp_min, self.cfg.clamp_max)
        if self._act_name == "sigmoid":
            return torch.sigmoid(x)
        x = self.softplus(x)
        return x.clamp(self.cfg.clamp_min, self.cfg.clamp_max)

    def forward(self, z: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            z: [B, d_in]
        Returns:
            f_hat: [B, K]
            c_hat: Optional[Tensor] [B, K] or None
        """
        # u: [B, D_t], V: [K, D_t]
        u = self.trunk(z)
        V = self.branch(self.lambdas01)  # [K, D_t]
        f_hat = u @ V.T                  # [B, K]
        f_hat = self._apply_activation(f_hat)

        c_hat: Optional[Tensor] = None
        if self.use_continuum:
            cw = self.cont_head(z)           # [B, M_c]
            c_hat = cw @ self.cont_basis.T   # [B, K]
            cmin, cmax = self.cfg.continuum_range
            span = (cmax - cmin) * 0.5
            mid = (cmax + cmin) * 0.5
            c_hat = (mid + span * torch.tanh(c_hat)).clamp(cmin, cmax)

        return f_hat, c_hat
