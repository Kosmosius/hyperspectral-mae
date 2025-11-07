"""Solid unmixing head."""
from __future__ import annotations

import torch
from torch import nn


class SimplexProjector(nn.Module):
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.softmax(logits, dim=-1)


class SolidsUnmixingHead(nn.Module):
    def __init__(self, embed_dim: int, num_endmembers: int):
        super().__init__()
        self.abundance_layer = nn.Linear(embed_dim, num_endmembers)
        self.projector = SimplexProjector()
        self.endmember_spectra = nn.Parameter(torch.randn(num_endmembers, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        abundances = self.projector(self.abundance_layer(x))
        reconstruction = abundances @ self.endmember_spectra
        return torch.cat([abundances, reconstruction], dim=-1)


__all__ = ["SolidsUnmixingHead"]
