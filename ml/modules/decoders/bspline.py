from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from torch import nn, Tensor

from ....ml.interfaces import DecoderProtocol


# --------------------- Utilities ---------------------

def _maybe_to_buffer(x: Tensor | "np.ndarray", device: torch.device, dtype: torch.dtype) -> Tensor:
    if not isinstance(x, Tensor):
        import numpy as np
        assert isinstance(x, np.ndarray)
        x = torch.from_numpy(x)
    return x.to(device=device, dtype=dtype)


def _make_poly_basis01(K: int, degree: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """
    Polynomial basis over normalized wavelength u in [0,1].
    Returns [K, degree+1] with columns [1, u, u^2, ...].
    """
    u = torch.linspace(0.0, 1.0, K, device=device, dtype=dtype)
    cols = [torch.ones_like(u)]
    for d in range(1, degree + 1):
        cols.append(u ** d)
    return torch.stack(cols, dim=-1)  # [K, degree+1]


# --------------------- Config ---------------------

@dataclass
class BSplineDecoderConfig:
    """
    B-spline decoder mapping z -> weights -> f_hat = sum_m w_m B_m(λ).

    Args:
        d_in:            latent dimension (z)
        bases:           list of basis matrices [K, M_b] (multi-scale allowed). These should be orthonormal (recommended).
        activation:      output activation for reflectance in [0,1] (identity|sigmoid|softplus_clamp)
        clamp_min/max:   range limits when using softplus_clamp (and final safety clamp in all modes)
        mlp_hidden:      hidden size for the z->coeffs MLP per-basis
        mlp_layers:      number of layers (>=1). If 1, it's a single Linear to coeffs.
        dropout:         dropout in hidden layers
        use_layernorm:   apply LayerNorm on z before heads (stabilizes training)
        use_continuum:   if True, also predict c_hat[K] using a coarse polynomial or provided basis
        continuum_degree:degree of polynomial (if continuum_basis is None)
        continuum_basis: optional custom continuum basis [K, M_c] (overrides polynomial)
        continuum_init_to_one: initialize continuum head close to 1.0
        continuum_range: allowed range for c_hat (min, max)
    """
    d_in: int
    bases: Sequence[Tensor]                          # one or more [K, M_b] tensors/arrays
    activation: str = "identity"                     # identity | sigmoid | softplus_clamp
    clamp_min: float = -0.05
    clamp_max: float = 1.05
    mlp_hidden: int = 512
    mlp_layers: int = 2
    dropout: float = 0.0
    use_layernorm: bool = True
    use_continuum: bool = True
    continuum_degree: int = 5
    continuum_basis: Optional[Tensor] = None         # [K, M_c]
    continuum_init_to_one: bool = True
    continuum_range: Tuple[float, float] = (0.7, 1.3)


# --------------------- Decoder ---------------------

class _HeadMLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        if layers <= 1:
            self.net = nn.Linear(d_in, d_out)
        else:
            blocks = [nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout)]
            for _ in range(layers - 2):
                blocks += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)]
            blocks += [nn.Linear(hidden, d_out)]
            self.net = nn.Sequential(*blocks)

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class BSplineDecoder(nn.Module, DecoderProtocol):
    """
    Multi-scale B-spline decoder with optional continuum head.
    """
    def __init__(self, cfg: BSplineDecoderConfig):
        super().__init__()
        self.cfg = cfg

        # Register bases as buffers (non-trainable)
        device = torch.device("cpu")
        dtype = torch.get_default_dtype()

        self.bases: list[Tensor] = []
        for B in cfg.bases:
            Bt = _maybe_to_buffer(B, device, dtype)  # [K, M]
            assert Bt.dim() == 2, "Each basis must be [K, M]"
            self.register_buffer(f"basis_{len(self.bases)}", Bt, persistent=False)
            self.bases.append(Bt)

        self.K = self.bases[0].shape[0]
        for b in self.bases[1:]:
            if b.shape[0] != self.K:
                raise ValueError("All bases must share same K")

        # Heads for coefficients (one per basis)
        self.norm = nn.LayerNorm(cfg.d_in) if cfg.use_layernorm else nn.Identity()
        self.heads = nn.ModuleList([
            _HeadMLP(cfg.d_in, B.shape[1], cfg.mlp_hidden, cfg.mlp_layers, cfg.dropout)
            for B in self.bases
        ])

        # Continuum basis/head
        self.use_continuum = cfg.use_continuum
        if self.use_continuum:
            if cfg.continuum_basis is not None:
                Cb = _maybe_to_buffer(cfg.continuum_basis, device, dtype)  # [K, M_c]
            else:
                Cb = _make_poly_basis01(self.K, cfg.continuum_degree, device, dtype)  # [K, M_c]
            self.register_buffer("cont_basis", Cb, persistent=False)
            self.cont_head = _HeadMLP(cfg.d_in, Cb.shape[1], max(64, cfg.mlp_hidden // 2), 2, cfg.dropout)

            # Initialize to predict near c_hat ≈ 1
            if cfg.continuum_init_to_one and isinstance(self.cont_head.net, nn.Sequential):
                last = self.cont_head.net[-1]
                if isinstance(last, nn.Linear):
                    nn.init.zeros_(last.weight)
                    nn.init.zeros_(last.bias)

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
        # softplus_clamp -> [0, +inf) then clamp to [clamp_min, clamp_max]
        x = self.softplus(x)  # >=0
        return x.clamp(self.cfg.clamp_min, self.cfg.clamp_max)

    def forward(self, z: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            z: [B, d_in]

        Returns:
            f_hat: [B, K]
            c_hat: Optional[Tensor] [B, K] or None
        """
        B = z.size(0)
        z_n = self.norm(z)

        # Predict coefficients per basis and sum contributions
        f_parts = []
        for head, Bmat in zip(self.heads, self.bases):
            coeffs = head(z_n)                     # [B, M_b]
            part = coeffs @ Bmat.T                 # [B, K]
            f_parts.append(part)
        f_hat = torch.stack(f_parts, dim=0).sum(dim=0)  # [B,K]
        f_hat = self._apply_activation(f_hat)

        c_hat: Optional[Tensor] = None
        if self.use_continuum:
            c_w = self.cont_head(z_n)              # [B, M_c]
            c_hat = c_w @ self.cont_basis.T        # [B, K]
            # Map to around 1: use tanh for bounded variation around 1
            c_min, c_max = self.cfg.continuum_range
            span = (c_max - c_min) * 0.5
            mid = (c_max + c_min) * 0.5
            c_hat = mid + span * torch.tanh(c_hat)
            c_hat = c_hat.clamp(c_min, c_max)

        return f_hat, c_hat
