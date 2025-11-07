"""Self-supervised pretraining stage."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from hsi_fm.model.backbone import HyperspectralMAE
from hsi_fm.model.losses import mae_loss, sam_loss


@dataclass
class StageAConfig:
    lr: float
    masking_spatial: float
    masking_spectral: float


class StageAExperiment:
    """Drives the self-supervised masked autoencoder stage."""

    def __init__(self, model: HyperspectralMAE, optimizer: torch.optim.Optimizer):
        self.model = model
        self.optimizer = optimizer

    def training_step(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        decoded = self.model(tokens)
        loss = mae_loss(decoded[:, 1:], targets) + sam_loss(decoded[:, 1:], targets)
        return loss

    def train_epoch(self, dataloader: Iterable[torch.Tensor]) -> float:
        total = 0.0
        steps = 0
        self.model.train()
        for batch in dataloader:
            tokens = batch["tokens"].to(next(self.model.parameters()).device)
            target = batch["target"].to(tokens.device)
            loss = self.training_step(tokens, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += float(loss.detach())
            steps += 1
        return total / max(steps, 1)


__all__ = ["StageAExperiment", "StageAConfig"]
