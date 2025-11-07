"""Transformer blocks with spectral-spatial factorization."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x


class FactorizedAttention(nn.Module):
    """Factorized spectral then spatial attention."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, spectral_groups: Optional[int] = None) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, n, self.num_heads, d // self.num_heads).transpose(1, 2)
        k = k.view(b, n, self.num_heads, d // self.num_heads).transpose(1, 2)
        v = v.view(b, n, self.num_heads, d // self.num_heads).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / (d ** 0.5)
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.proj(out)


class FactorizedBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FactorizedAttention(dim, num_heads)
        self.scale1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.scale2 = LayerScale(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.scale1(self.attn(self.norm1(x))))
        x = x + self.drop_path(self.scale2(self.mlp(self.norm2(x))))
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1 - self.drop_prob
        mask = torch.rand(x.shape[0], 1, 1, device=x.device, dtype=x.dtype) < keep
        return x * mask / keep


__all__ = ["FactorizedBlock", "LayerScale", "DropPath"]
