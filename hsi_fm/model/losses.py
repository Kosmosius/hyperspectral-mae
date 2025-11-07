"""Collection of losses for the hyperspectral foundation model."""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


def mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.l1_loss(pred, target)


def sam_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_norm = nn.functional.normalize(pred, dim=-1)
    target_norm = nn.functional.normalize(target, dim=-1)
    cosine = (pred_norm * target_norm).sum(dim=-1).clamp(-1.0, 1.0)
    return 1.0 - cosine.mean()


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1 - (2 * intersection + eps) / (union + eps)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute the mean squared error over the valid locations of ``mask``."""

    mask_bool = mask.to(dtype=torch.bool)
    diff = (pred - target) ** 2
    if mask_bool.any():
        return diff[mask_bool].mean()
    return torch.tensor(0.0, dtype=pred.dtype, device=pred.device)


def info_nce(queries: torch.Tensor, keys: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Contrastive loss encouraging matching query/key pairs."""

    queries_norm = nn.functional.normalize(queries, dim=-1)
    keys_norm = nn.functional.normalize(keys, dim=-1)
    logits = queries_norm @ keys_norm.transpose(0, 1)
    logits = logits / max(temperature, 1e-6)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return nn.functional.cross_entropy(logits, labels)


def cycle_loss(original: torch.Tensor, cycled: torch.Tensor) -> torch.Tensor:
    """Penalize reconstruction error after a cycle-consistency pass."""

    return nn.functional.mse_loss(cycled, original)


__all__ = [
    "mae_loss",
    "sam_loss",
    "dice_loss",
    "masked_mse",
    "info_nce",
    "cycle_loss",
]
