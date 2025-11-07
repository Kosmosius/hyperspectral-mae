# hsi_fm/model/losses.py
"""Collection of losses for the hyperspectral foundation model."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


__all__ = [
    "mae_loss",
    "sam_loss",
    "dice_loss",
    "masked_mse",
    "info_nce",
    "cycle_loss",
]


# ---------------------------------------------------------------------
# Basic reconstruction losses
# ---------------------------------------------------------------------

def mae_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean absolute error (L1)."""
    return F.l1_loss(pred, target)


def masked_mse(pred: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """
    Mean squared error over valid locations.

    Args:
        pred:   (..., D)
        target: (..., D)
        mask:   (...,) or (..., 1) or (..., D) boolean/float. True/1 means 'keep'.
    """
    diff = (pred - target) ** 2
    if mask is None:
        return diff.mean()

    m = mask
    # Coerce to boolean and broadcast across the last dim if needed
    if m.dtype != torch.bool:
        m = m != 0
    while m.ndim < diff.ndim:
        m = m.unsqueeze(-1)
    if m.shape != diff.shape:
        # allow trailing singleton or last-dim broadcast
        if m.shape[-1] == 1:
            m = m.expand_as(diff)
        elif m.shape[:-1] == diff.shape[:-1]:
            m = m.expand_as(diff)
        else:
            raise ValueError(f"Incompatible mask shape {m.shape} for diff {diff.shape}")

    sel = diff.masked_select(m)
    if sel.numel() == 0:
        return torch.zeros((), dtype=pred.dtype, device=pred.device)
    return sel.mean()


# ---------------------------------------------------------------------
# Spectral losses
# ---------------------------------------------------------------------

def sam_loss(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Spectral Angle Mapper (SAM) in radians, averaged over batch.

    Args:
        pred:   (..., L)
        target: (..., L)
    Returns:
        Mean spectral angle in radians (lower is better).
    """
    # Flatten leading dims into batch for clarity
    p = pred.reshape(-1, pred.shape[-1])
    t = target.reshape(-1, target.shape[-1])
    num = (p * t).sum(dim=-1)
    den = (p.norm(dim=-1) * t.norm(dim=-1)).clamp_min(eps)
    cos = (num / den).clamp(-1.0, 1.0)
    angle = torch.arccos(cos)
    return angle.mean()


# ---------------------------------------------------------------------
# Segmentation-style loss (kept, generalized)
# ---------------------------------------------------------------------

def dice_loss(logits: Tensor, targets: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Binary Dice loss on NCHW (or any tensor ≥ 2D). Applies sigmoid to logits.
    """
    probs = torch.sigmoid(logits)
    # Flatten all but batch
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * targets).sum(dim=dims)
    union = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


# ---------------------------------------------------------------------
# Contrastive & cycle
# ---------------------------------------------------------------------

def info_nce(queries: Tensor, keys: Tensor, temperature: float = 0.1) -> Tensor:
    """
    NT-Xent / InfoNCE with identity positives (index-aligned).
    If B_q != B_k, the shorter batch dimension is used.
    """
    assert queries.ndim == 2 and keys.ndim == 2, "Expected (B,d) embeddings."
    B = min(queries.size(0), keys.size(0))
    q = F.normalize(queries[:B], dim=-1)
    k = F.normalize(keys[:B], dim=-1)
    logits = (q @ k.t()) / max(temperature, 1e-6)  # (B,B)
    labels = torch.arange(B, device=logits.device)
    return F.cross_entropy(logits, labels)


def cycle_loss(
    original: Tensor,
    cycled: Tensor,
    mask: Optional[Tensor] = None,
    use_sam: bool = False,
    sam_weight: float = 0.5,
) -> Tensor:
    """
    Penalize reconstruction error after a cycle-consistency pass.

    Combines masked MSE with optional SAM on the spectral axis.

    Args:
        original: (…, L)
        cycled:   (…, L)
        mask:     optional mask broadcastable to inputs
        use_sam:  include SAM term
        sam_weight: weight applied to SAM when used (0..1). Final loss is:
                    (1 - sam_weight) * MSE + sam_weight * SAM
    """
    mse = masked_mse(cycled, original, mask)
    if not use_sam:
        return mse
    # SAM doesn't need the mask (spectral vector-wise); we compute on all valid rows
    sam = sam_loss(cycled, original)
    # convex combination (defaults to 0.5 if you turn it on)
    return (1.0 - sam_weight) * mse + sam_weight * sam
