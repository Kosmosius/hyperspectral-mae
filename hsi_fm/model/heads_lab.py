"""Heads for lab spectra supervision."""
from __future__ import annotations

import torch
from torch import nn


class LabPrototypeHead(nn.Module):
    def __init__(self, embed_dim: int, num_prototypes: int):
        super().__init__()
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.normalize(self.proj(x), dim=-1)
        protos = nn.functional.normalize(self.prototypes, dim=-1)
        return x @ protos.t()


__all__ = ["LabPrototypeHead"]
