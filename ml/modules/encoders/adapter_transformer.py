from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
from torch import nn, Tensor
import math

from ....ml.interfaces import EncoderOutput
from .relpos_bias import RelPosRFFBias, RelPosRFFBiasConfig


# ----------------- utilities -----------------

class LayerScale(nn.Module):
    """
    LayerScale: learnable per-channel residual scaling (Touvron et al., 2021).
    """
    def __init__(self, dim: int, init: float = 1e-4):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init)

    def forward(self, x: Tensor) -> Tensor:
        # x: [..., dim]
        return x * self.gamma


def attention_mask_to_logits(mask: Optional[Tensor], T: int, device, dtype) -> Optional[Tensor]:
    """
    Convert [B,T] boolean mask (True=keep) to additive attention mask logits [B,1,T,T]
    with -inf where either query or key is padding.
    """
    if mask is None:
        return None
    m = mask
    attn = m[:, None, :, None] & m[:, None, None, :]  # [B,1,T,T]
    add = torch.zeros_like(attn, dtype=dtype, device=device)
    add.masked_fill_(~attn, float("-inf"))
    return add


class MultiheadAttentionWithBias(nn.Module):
    """
    Minimal MHA that accepts an external additive bias [B,H,T,T].
    """
    def __init__(self, embed_dim: int, num_heads: int, attn_drop: float, proj_drop: float):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, mask_logits: Optional[Tensor] = None, bias: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x: [B,T,C]
            mask_logits: [B,1,T,T] with -inf at masked positions (optional)
            bias: [B,H,T,T] additive bias to logits (optional)

        Returns:
            y: [B,T,C]
            attn: [B,H,T,T] (optional; detached for debug)
        """
        B, T, C = x.shape
        qkv = self.qkv(x)  # [B,T,3C]
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # 3, B, H, T, Dh
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,H,T,Dh]

        attn_logits = (q @ k.transpose(-2, -1)) * self.scale  # [B,H,T,T]
        if bias is not None:
            attn_logits = attn_logits + bias
        if mask_logits is not None:
            attn_logits = attn_logits + mask_logits

        attn = attn_logits.softmax(dim=-1)  # [B,H,T,T]
        attn = self.attn_drop(attn)
        y = attn @ v  # [B,H,T,Dh]
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # [B,T,C]
        y = self.proj_drop(self.proj(y))
        return y, attn.detach()


class PreLNTransformerBlock(nn.Module):
    """
    Pre-LN Transformer block with LayerScale and support for external attention bias.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        drop: float = 0.0,
        ls_init: float = 1e-4,
        act: str = "gelu",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiheadAttentionWithBias(dim, num_heads, attn_drop, drop)
        self.ls1 = LayerScale(dim, init=ls_init)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU() if act == "gelu" else nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )
        self.ls2 = LayerScale(dim, init=ls_init)

    def forward(self, x: Tensor, mask_logits: Optional[Tensor], bias: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        y, attn = self.attn(self.norm1(x), mask_logits, bias)
        x = x + self.ls1(y)
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x, attn


# ----------------- encoder -----------------

@dataclass
class AdapterTransformerConfig:
    """
    Adapter-Transformer encoder for spectral band tokens.

    Args:
        d_in:           input token dim (USR token size)
        d_model:        transformer width
        depth:          number of Transformer blocks
        num_heads:      attention heads
        adapter_channels: conv adapter channels (pre-encoder denoising / local aggregation)
        adapter_kernels: list of kernel sizes for Conv1d stack (odd ints)
        adapter_strides: list of strides for Conv1d stack (same length as kernels)
        dropout:        dropout in MLP/proj
        attn_drop:      attention dropout
        ls_init:        LayerScale init
        pool:           'mean' or 'cls'
        use_relpos:     enable RFF relative bias over |Δλ|
        relpos_cfg:     config for RelPosRFFBias
    """
    d_in: int
    d_model: int = 512
    depth: int = 6
    num_heads: int = 8
    adapter_channels: int = 128
    adapter_kernels: tuple[int, ...] = (7, 7, 5)
    adapter_strides: tuple[int, ...] = (1, 1, 1)
    dropout: float = 0.1
    attn_drop: float = 0.0
    ls_init: float = 1e-4
    pool: str = "mean"  # or 'cls'
    use_relpos: bool = True
    relpos_cfg: Optional[RelPosRFFBiasConfig] = None
    track_attn: bool = False


class AdapterTransformerEncoder(nn.Module):
    """
    Adapter-Transformer encoder:
      - (optional) 1D Conv adapter over token dimension for local spectral context
      - Pre-LN Transformer blocks with optional relative position bias (|Δλ|)
      - Pooled latent z via mean or CLS
    """
    def __init__(self, cfg: AdapterTransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.input_proj = nn.Linear(cfg.d_in, cfg.d_model)

        # Adapter conv stack expects [B, C, T]; we apply on the projected features
        convs: List[nn.Module] = []
        C = cfg.d_model
        in_ch = C
        for k, s in zip(cfg.adapter_kernels, cfg.adapter_strides):
            pad = (k - 1) // 2
            convs += [nn.Conv1d(in_ch, cfg.adapter_channels, kernel_size=k, stride=s, padding=pad),
                      nn.BatchNorm1d(cfg.adapter_channels),
                      nn.ReLU(inplace=True)]
            in_ch = cfg.adapter_channels
        convs += [nn.Conv1d(in_ch, C, kernel_size=1)]
        self.adapter = nn.Sequential(*convs)

        # Transformer blocks
        blocks = []
        for _ in range(cfg.depth):
            blocks.append(
                PreLNTransformerBlock(
                    dim=cfg.d_model,
                    num_heads=cfg.num_heads,
                    mlp_ratio=4.0,
                    attn_drop=cfg.attn_drop,
                    drop=cfg.dropout,
                    ls_init=cfg.ls_init,
                    act="gelu",
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(cfg.d_model)

        # Optional CLS token
        if cfg.pool == "cls":
            self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        else:
            self.register_parameter("cls", None)

        # Rel-pos bias
        if cfg.use_relpos:
            rel_cfg = cfg.relpos_cfg or RelPosRFFBiasConfig(num_heads=cfg.num_heads)
            self.rel_bias = RelPosRFFBias(rel_cfg)
        else:
            self.rel_bias = None

        self.track_attn = cfg.track_attn

    def forward(
        self,
        tokens: Tensor,                    # [B,T,D_in]
        mask: Optional[Tensor] = None,     # [B,T] True=keep
        centers_nm: Optional[Tensor] = None,
        centers01: Optional[Tensor] = None,
    ) -> EncoderOutput:
        """
        Returns:
            EncoderOutput with z [B,d_model], tokens_out [B,T',d_model], optional attn maps.
        """
        B, T, _ = tokens.shape
        x = self.input_proj(tokens)  # [B,T,d_model]

        # Prepare mask
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=tokens.device)

        # Adapter conv expects [B,C,T]
        x_adapter = self.adapter(x.transpose(1, 2)).transpose(1, 2)  # [B,T,d_model]

        # Append CLS if needed
        if self.cfg.pool == "cls":
            cls_tok = self.cls.expand(B, -1, -1)  # [B,1,d_model]
            x_adapter = torch.cat([cls_tok, x_adapter], dim=1)
            # mask: pad (keep cls=True)
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)
            # centers need a placeholder for CLS if rel-bias is on
            if centers01 is not None:
                zeros = torch.zeros(B, 1, device=centers01.device, dtype=centers01.dtype)
                centers01 = torch.cat([zeros, centers01], dim=1)
            if centers_nm is not None and centers01 is None:
                zeros = torch.zeros(B, 1, device=centers_nm.device, dtype=centers_nm.dtype)
                centers_nm = torch.cat([zeros, centers_nm], dim=1)

        # Attention masks
        T2 = x_adapter.shape[1]
        attn_mask_logits = attention_mask_to_logits(mask, T2, device=x_adapter.device, dtype=x_adapter.dtype)

        # Precompute relative bias if enabled
        bias = None
        if self.rel_bias is not None:
            bias = self.rel_bias(centers_nm=centers_nm, centers01=centers01, mask=mask)  # [B,H,T2,T2]

        attn_maps: List[Tensor] = []
        x_out = x_adapter
        for blk in self.blocks:
            x_out, attn = blk(x_out, attn_mask_logits, bias)
            if self.track_attn:
                attn_maps.append(attn)

        x_out = self.norm(x_out)  # [B,T2,d_model]

        # Pool
        if self.cfg.pool == "cls":
            z = x_out[:, 0, :]  # [B,d_model]
            tokens_out = x_out[:, 1:, :]
            mask_out = mask[:, 1:]
        else:
            # masked mean
            m = mask.float().unsqueeze(-1)  # [B,T2,1]
            z = (x_out * m).sum(dim=1) / (m.sum(dim=1).clamp_min(1e-6))
            tokens_out = x_out
            mask_out = mask

        return EncoderOutput(
            z=z,
            tokens_out=tokens_out,
            attn_maps=attn_maps if self.track_attn else None,
            mask=mask_out,
        )
