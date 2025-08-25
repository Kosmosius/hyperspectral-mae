from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
from torch import nn, Tensor

from ....ml.interfaces import EncoderOutput
from .relpos_bias import RelPosRFFBias, RelPosRFFBiasConfig
from .adapter_transformer import (
    MultiheadAttentionWithBias,
    LayerScale,
    attention_mask_to_logits,
)


# ---------- Set Attention Blocks ----------

class SAB(nn.Module):
    """
    Set Attention Block (Lee et al., 2019) with Pre-LN and attention bias support.
    """
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 2.0, drop: float = 0.0, attn_drop: float = 0.0, ls_init: float = 1e-4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiheadAttentionWithBias(dim, heads, attn_drop, drop)
        self.ls1 = LayerScale(dim, init=ls_init)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )
        self.ls2 = LayerScale(dim, init=ls_init)

    def forward(self, X: Tensor, mask_logits: Optional[Tensor], bias: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        y, attn = self.attn(self.norm1(X), mask_logits, bias)
        X = X + self.ls1(y)
        X = X + self.ls2(self.ff(self.norm2(X)))
        return X, attn


class PMA(nn.Module):
    """
    Pooling by Multihead Attention with learnable seed vectors.
    Returns S pooled vectors (default S=1).
    """
    def __init__(self, dim: int, heads: int, S: int = 1, drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.S = S
        self.seed = nn.Parameter(torch.randn(1, S, dim) * 0.02)
        self.attn = MultiheadAttentionWithBias(dim, heads, attn_drop, drop)
        self.norm = nn.LayerNorm(dim)

    def forward(self, X: Tensor, mask_logits: Optional[Tensor], bias: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        B = X.size(0)
        Q = self.seed.expand(B, -1, -1)  # [B,S,dim]
        # "Cross-attention": queries = seeds, keys=values=X
        # Reuse module by concatenating and slicing:
        # We build a custom path: compute attention logits from Q and K separately would need a different module;
        # simpler: emulate by passing concatenated [Q; X] then extracting first S queries.
        # For clarity, implement lightweight attention inline here:

        # Project Q, K, V with the same linear for shape consistency
        # We’ll borrow internals from MultiheadAttentionWithBias by calling its qkv on a concat and slicing.
        B_, T, C = X.shape
        concat = torch.cat([Q, X], dim=1)  # [B, S+T, C]
        qkv = self.attn.qkv(concat)  # [B, S+T, 3C]
        qkv = qkv.view(B_, self.S + T, 3, self.attn.num_heads, self.attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,H,S+T,Dh]
        q = q[:, :, : self.S, :]          # [B,H,S,Dh]
        k = k[:, :, self.S :, :]          # [B,H,T,Dh]
        v = v[:, :, self.S :, :]

        logits = (q @ k.transpose(-2, -1)) * self.attn.scale  # [B,H,S,T]
        if bias is not None:
            # bias is [B,H,T,T]; for Q=seeds we don't have center positions; skip bias in PMA
            pass
        if mask_logits is not None:
            # reduce to [B,1,S,T]
            mask_k = mask_logits[:, :, 0:1, :]  # broadcast pattern; not exact, but prevents attending to pads
            logits = logits + mask_k

        attn = logits.softmax(dim=-1)  # [B,H,S,T]
        attn = self.attn.attn_drop(attn)
        Y = attn @ v  # [B,H,S,Dh]
        Y = Y.transpose(1, 2).contiguous().view(B_, self.S, C)
        out = self.attn.proj_drop(self.attn.proj(Y))
        out = self.norm(out)
        return out, attn


# ---------- Encoder ----------

@dataclass
class SetTransformerConfig:
    """
    SetTransformer encoder with SAB stacks and PMA pooling.

    Args:
        d_in:        input token dimension
        d_model:     internal width
        depth:       number of SAB blocks
        num_heads:   attention heads
        mlp_ratio:   MLP expansion ratio in SAB
        dropout:     dropout in MLP/proj
        attn_drop:   attention dropout
        ls_init:     LayerScale init
        pool_S:      number of pooled seeds (S=1 -> single z; >1 returns average as z and keeps S tokens)
        use_relpos:  enable relative positional bias in SAB blocks
        relpos_cfg:  config for RelPosRFFBias
        track_attn:  keep attention maps (debug)
    """
    d_in: int
    d_model: int = 512
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    attn_drop: float = 0.0
    ls_init: float = 1e-4
    pool_S: int = 1
    use_relpos: bool = True
    relpos_cfg: Optional[RelPosRFFBiasConfig] = None
    track_attn: bool = False


class SetTransformerEncoder(nn.Module):
    """
    Permutation-invariant encoder using SAB (+ optional rel-pos bias) and PMA pooling.
    """
    def __init__(self, cfg: SetTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.d_in, cfg.d_model)

        self.sabs = nn.ModuleList([
            SAB(cfg.d_model, cfg.num_heads, cfg.mlp_ratio, cfg.dropout, cfg.attn_drop, cfg.ls_init)
            for _ in range(cfg.depth)
        ])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.pma = PMA(cfg.d_model, cfg.num_heads, S=cfg.pool_S, drop=cfg.dropout, attn_drop=cfg.attn_drop)

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
        B, T, _ = tokens.shape
        x = self.in_proj(tokens)  # [B,T,d_model]

        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=tokens.device)

        mask_logits = attention_mask_to_logits(mask, T, device=x.device, dtype=x.dtype)

        bias = None
        if self.rel_bias is not None:
            bias = self.rel_bias(centers_nm=centers_nm, centers01=centers01, mask=mask)  # [B,H,T,T]

        attn_maps: List[Tensor] = []
        for sab in self.sabs:
            x, attn = sab(x, mask_logits, bias)
            if self.track_attn:
                attn_maps.append(attn)

        x = self.norm(x)

        pooled, attn_pma = self.pma(x, mask_logits, bias)  # [B,S,d_model]
        if self.track_attn:
            # PMA attention is over seeds; optional to append
            pass

        if self.cfg.pool_S == 1:
            z = pooled[:, 0, :]
            tokens_out = x
        else:
            # Aggregate S pooled vectors into a single z (mean); also return x as tokens_out
            z = pooled.mean(dim=1)
            tokens_out = x

        return EncoderOutput(
            z=z,
            tokens_out=tokens_out,
            attn_maps=attn_maps if self.track_attn else None,
            mask=mask,
        )
