"""Patch embedding layers for hyperspectral cubes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


@dataclass
class TubePatchConfig:
    patch_size: Tuple[int, int]
    stride: Tuple[int, int]


class TubePatchEmbed3D(nn.Module):
    """Embed spectral tubes using a 3D convolution."""

    def __init__(self, in_channels: int, embed_dim: int, patch_config: TubePatchConfig):
        super().__init__()
        ph, pw = patch_config.patch_size
        sh, sw = patch_config.stride
        self.patch_config = patch_config
        self.proj = nn.Conv3d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=(in_channels, ph, pw),
            stride=(in_channels, sh, sw),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``x`` into embeddings.

        Parameters
        ----------
        x:
            Tensor shaped ``(B, C, H, W)``.
        """

        if x.ndim != 4:
            raise ValueError("Expected input shaped (B, C, H, W)")
        b, c, h, w = x.shape
        x = x.unsqueeze(1)  # (B, 1, C, H, W)
        if c != self.proj.kernel_size[0]:
            raise ValueError("Spectral size mismatch with patch embedding")
        patches = self.proj(x)
        patches = patches.flatten(2).transpose(1, 2)
        return patches


class SpectralPositionalEncoding(nn.Module):
    """Fourier positional encodings along the spectral axis."""

    def __init__(self, num_bands: int, embed_dim: int):
        super().__init__()
        self.num_bands = num_bands
        self.embed_dim = embed_dim
        freqs = torch.arange(embed_dim // 2, dtype=torch.float32)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, wavelengths: torch.Tensor) -> torch.Tensor:
        wavelengths = wavelengths.unsqueeze(-1)
        phases = 2 * torch.pi * wavelengths * self.freqs
        sin = torch.sin(phases)
        cos = torch.cos(phases)
        return torch.cat([sin, cos], dim=-1)


__all__ = ["TubePatchConfig", "TubePatchEmbed3D", "SpectralPositionalEncoding"]
