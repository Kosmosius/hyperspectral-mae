from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn, Tensor
import math


@dataclass
class RelPosRFFBiasConfig:
    """
    Relative position bias using Random Fourier Features over |Δλ|.

    - If centers01 provided: interpret as normalized [0,1]; freq range applies directly.
    - If centers_nm provided: you should set wavelength_min_nm/max_nm to normalize.

    The module outputs an additive bias tensor shaped [B, H, T, T] that is added to
    attention logits before softmax: logits = (QK^T) / sqrt(d_h) + bias.
    """
    num_heads: int
    rff_dim: int = 16                 # number of frequencies (each -> sin/cos => 2*rff_dim features)
    f_min: float = 2.0                # min frequency (cycles over [0,1])
    f_max: float = 64.0               # max frequency
    use_squared_dist: bool = False    # optionally encode |Δ|^2 to emphasize locality
    hidden_dim: int = 64              # MLP hidden size to map RFF -> head biases
    act: str = "gelu"
    bias_scale: float = 1.0           # global scale applied to outputs
    wavelength_min_nm: float = 400.0  # if normalizing nm -> [0,1]
    wavelength_max_nm: float = 2500.0
    dropout: float = 0.0
    seed: int = 1234


def _act(name: str) -> nn.Module:
    if name.lower() == "gelu":
        return nn.GELU()
    if name.lower() == "relu":
        return nn.ReLU(inplace=True)
    if name.lower() == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class RelPosRFFBias(nn.Module):
    """
    Compute an additive attention bias over pairs (i,j) using an RFF of |Δλ|.

    Steps:
      - normalize centers to [0,1] (if nm provided)
      - D = |c_i - c_j|  ->  Φ(D) = [sin(2π f_k D + b_k), cos(...)]
      - small MLP maps Φ(D) -> per-head scalar; broadcast to [B,H,T,T]
    """
    def __init__(self, cfg: RelPosRFFBiasConfig):
        super().__init__()
        self.cfg = cfg

        torch.manual_seed(cfg.seed)
        # Log-spaced frequencies
        freqs = torch.logspace(
            math.log10(cfg.f_min), math.log10(cfg.f_max), cfg.rff_dim
        )  # [M]
        self.register_buffer("freqs", freqs, persistent=False)
        self.bias_phase = nn.Parameter(torch.rand(cfg.rff_dim) * (2 * math.pi))
        in_dim = 2 * cfg.rff_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            _act(cfg.act),
            nn.Linear(cfg.hidden_dim, cfg.num_heads),
        )
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(
        self,
        *,
        centers_nm: Optional[Tensor] = None,   # [B,T]
        centers01: Optional[Tensor] = None,    # [B,T]
        mask: Optional[Tensor] = None,         # [B,T] True=keep, False=pad
    ) -> Tensor:
        """
        Returns:
            bias: [B, H, T, T] additive attention bias (masked positions set to 0)
        """
        if centers01 is None and centers_nm is None:
            raise ValueError("Provide centers01 or centers_nm")

        if centers01 is None:
            # normalize nm -> [0,1]
            cmin = self.cfg.wavelength_min_nm
            cmax = self.cfg.wavelength_max_nm
            centers01 = (centers_nm - cmin) / max(1e-9, (cmax - cmin))
            centers01 = centers01.clamp(0.0, 1.0)

        # centers01: [B,T] -> pairwise distances [B,T,T]
        D = (centers01.unsqueeze(-1) - centers01.unsqueeze(-2)).abs()
        if self.cfg.use_squared_dist:
            D = D * D

        # RFF: sin/cos on D
        # [B,T,T,1] * [M] -> [B,T,T,M]
        arg = 2 * math.pi * D.unsqueeze(-1) * self.freqs + self.bias_phase
        sin = torch.sin(arg)
        cos = torch.cos(arg)
        feats = torch.cat([sin, cos], dim=-1)  # [B,T,T,2M]

        # Map to per-head bias
        B_, T, _, _ = feats.shape[0], feats.shape[1], feats.shape[2], feats.shape[3]
        H = self.cfg.num_heads

        # Flatten the last two dims for a single MLP pass
        x = feats.view(B_ * T * T, -1)
        b = self.mlp(x)  # [B*T*T, H]
        b = b.view(B_, T, T, H).permute(0, 3, 1, 2).contiguous()  # [B,H,T,T]
        b = self.dropout(b)
        b = b * self.cfg.bias_scale

        if mask is not None:
            # mask: True=keep; attention should not see False positions
            # We'll zero out biases to/from padding; actual logits masking happens in the attention module.
            m = mask[:, None, :, None] & mask[:, None, None, :]  # [B,1,T,1] & [B,1,1,T] -> [B,1,T,T]
            b = b * m

        return b
