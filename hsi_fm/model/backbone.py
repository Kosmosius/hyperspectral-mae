"""Shared spectral-spatial masked autoencoder backbone."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .blocks import FactorizedBlock


@dataclass
class MaskingConfig:
    spatial_ratio: float
    spectral_ratio: float


class HyperspectralMAE(nn.Module):
    """Simplified MAE backbone suitable for unit tests and integration."""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        masking: Optional[MaskingConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.masking = masking or MaskingConfig(0.5, 0.5)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, embed_dim))
        self.blocks = nn.ModuleList(
            [FactorizedBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, n, d = tokens.shape
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        pos = self.pos_embed[:, : n + 1]
        x = x + pos
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        decoded = self.decoder(x)
        return decoded

    def compute_mask(self, num_tokens: int, device: torch.device) -> torch.Tensor:
        """Generate boolean mask for tokens."""

        num_mask = int(num_tokens * self.masking.spectral_ratio)
        mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        if num_mask > 0:
            perm = torch.randperm(num_tokens, device=device)[:num_mask]
            mask[perm] = True
        return mask


__all__ = ["HyperspectralMAE", "MaskingConfig"]
