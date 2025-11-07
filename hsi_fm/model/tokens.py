"""Embedding helpers for modality and sensor tokens."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class SensorEmbedding(nn.Module):
    """Lookup embeddings for sensor identifiers."""

    def __init__(self, sensors: Dict[str, int], embed_dim: int):
        super().__init__()
        self.sensors = sensors
        self.embed = nn.Embedding(len(sensors), embed_dim)

    def forward(self, sensor: str) -> torch.Tensor:
        if sensor not in self.sensors:
            raise KeyError(f"Unknown sensor: {sensor}")
        idx = torch.tensor([self.sensors[sensor]], dtype=torch.long, device=self.embed.weight.device)
        return self.embed(idx).squeeze(0)


class GeometryEmbedding(nn.Module):
    """Small MLP encoding observation geometry metadata."""

    def __init__(self, in_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        return self.net(metadata)


__all__ = ["SensorEmbedding", "GeometryEmbedding"]
