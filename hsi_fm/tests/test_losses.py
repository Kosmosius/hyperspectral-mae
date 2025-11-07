"""Tests for loss helpers."""
from __future__ import annotations

import torch

from hsi_fm.model.losses import cycle_loss, info_nce, masked_mse


def test_masked_mse_respects_mask() -> None:
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.zeros_like(pred)
    mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss = masked_mse(pred, target, mask.bool())
    expected = (pred[0, 0] ** 2 + pred[1, 1] ** 2) / 2
    assert torch.allclose(loss, expected)


def test_info_nce_prefers_matching_pairs() -> None:
    queries = torch.eye(2)
    keys_same = torch.eye(2)
    keys_swap = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    loss_same = info_nce(queries, keys_same, temperature=0.1)
    loss_swap = info_nce(queries, keys_swap, temperature=0.1)
    assert loss_same < loss_swap


def test_cycle_loss_increases_with_error() -> None:
    original = torch.ones(2, 4)
    cycled = original.clone()
    loss_perfect = cycle_loss(original, cycled)
    loss_worse = cycle_loss(original, cycled + 0.2)
    assert loss_perfect < loss_worse
