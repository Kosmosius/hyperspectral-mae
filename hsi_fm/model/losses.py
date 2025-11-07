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


__all__ = ["mae_loss", "sam_loss", "dice_loss"]
