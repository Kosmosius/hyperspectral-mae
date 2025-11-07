"""Collection of losses for the hyperspectral foundation model."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    "mae_loss",
    "sam_loss",
    "dice_loss",
    "masked_mse",
    "info_nce",
    "cycle_loss",
]


def mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def sam_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred_norm = F.normalize(pred, dim=-1, eps=eps)
    target_norm = F.normalize(target, dim=-1, eps=eps)
    cosine = (pred_norm * target_norm).sum(dim=-1).clamp(-1.0, 1.0)
    return 1.0 - cosine.mean()


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1 - (2 * intersection + eps) / (union + eps)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean squared error that ignores masked elements."""

    diff = (pred - target) ** 2
    if mask is not None:
        mask_f = mask.to(dtype=diff.dtype)
        diff = diff * mask_f
        denom = mask_f.sum().clamp_min(1.0)
    else:
        denom = diff.new_tensor(diff.numel(), dtype=diff.dtype)
    return diff.sum() / denom


def info_nce(
    queries: torch.Tensor,
    keys: torch.Tensor,
    temperature: float,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """NT-Xent loss with optional masking of invalid pairs."""

    if queries.shape != keys.shape:
        raise ValueError("queries and keys must have the same shape")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    q = F.normalize(queries, dim=-1)
    k = F.normalize(keys, dim=-1)
    logits = q @ k.T / temperature
    if mask is not None:
        if mask.shape != queries.shape[:1]:
            raise ValueError("mask must have shape (batch,)")
        valid = mask.to(dtype=torch.bool)
        q = q[valid]
        k = k[valid]
        logits = q @ k.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = F.cross_entropy(logits, labels)
    return loss


def cycle_loss(
    original: torch.Tensor,
    cycled: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cycle consistency using MSE + optional SAM for spectra."""

    mse = masked_mse(cycled, original, mask)
    if original.shape[-1] > 1:
        sam = sam_loss(cycled, original)
    else:
        sam = torch.tensor(0.0, device=original.device, dtype=original.dtype)
    return mse + sam
