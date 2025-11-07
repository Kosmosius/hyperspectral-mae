"""Methane plume evaluation utilities."""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch


def compute_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    intersection = torch.logical_and(pred_mask.bool(), gt_mask.bool()).sum()
    union = torch.logical_or(pred_mask.bool(), gt_mask.bool()).sum()
    if union == 0:
        return 1.0
    return float((intersection / union).item())


def mdl_curve(detections: Sequence[bool], emission_rates: Sequence[float]) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(detections) != len(emission_rates):
        raise ValueError("Length mismatch between detections and emission rates")
    rates = torch.tensor(emission_rates, dtype=torch.float32)
    det = torch.tensor(detections, dtype=torch.float32)
    order = torch.argsort(rates)
    rates_sorted = rates[order]
    det_sorted = det[order]
    cumulative = torch.cumsum(det_sorted, dim=0) / (torch.arange(len(det_sorted), dtype=torch.float32) + 1)
    return rates_sorted, cumulative


def evaluate_methane_mdl(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> Dict[str, float]:
    ious = [compute_iou(p, g) for p, g in zip(pred_masks, gt_masks)]
    return {"iou": float(sum(ious) / max(len(ious), 1))}


__all__ = ["evaluate_methane_mdl", "compute_iou", "mdl_curve"]
