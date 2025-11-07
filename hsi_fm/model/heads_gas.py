"""Gas plume detection head."""
from __future__ import annotations

import torch
from torch import nn


class GasPlumeHead(nn.Module):
    def __init__(self, embed_dim: int, num_gases: int):
        super().__init__()
        self.background = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
        )
        self.segmentation = nn.Conv2d(embed_dim, num_gases, 1)
        self.column = nn.Conv2d(embed_dim, num_gases, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        bg = self.background(features)
        seg = self.segmentation(bg)
        column = self.column(bg)
        return torch.cat([seg, column], dim=1)


__all__ = ["GasPlumeHead"]
