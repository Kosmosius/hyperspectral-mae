"""EMIT mineral benchmark evaluation."""
from __future__ import annotations

from typing import Dict

import torch


def evaluate_emit_min(pred_abundances: torch.Tensor, gt_abundances: torch.Tensor) -> Dict[str, float]:
    if pred_abundances.shape != gt_abundances.shape:
        raise ValueError("Shape mismatch between predictions and ground truth")
    eps = 1e-6
    tp = torch.minimum(pred_abundances, gt_abundances).sum()
    precision = tp / (pred_abundances.sum() + eps)
    recall = tp / (gt_abundances.sum() + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {"f1": float(f1), "precision": float(precision), "recall": float(recall)}


__all__ = ["evaluate_emit_min"]
