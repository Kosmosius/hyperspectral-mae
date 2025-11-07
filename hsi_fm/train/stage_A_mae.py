"""Self-supervised pretraining stage."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import torch
from torch import nn

from hsi_fm.model.backbone import HyperspectralMAE
from hsi_fm.model.losses import mae_loss, sam_loss
from hsi_fm.train.ema import EMA


@dataclass
class StageAConfig:
    """Configuration for Stage A training."""

    lr: float = 3e-4
    masking_spatial: float = 0.6
    masking_spectral: float = 0.3


class StageAExperiment:
    """Drives the self-supervised masked autoencoder stage."""

    def __init__(
        self,
        model: HyperspectralMAE,
        optimizer: torch.optim.Optimizer,
        ema: Optional[EMA] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.ema = ema

    def training_step(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        tokens = batch["tokens"].to(next(self.model.parameters()).device)
        target = batch["target"].to(tokens.device)
        decoded = self.model(tokens)
        loss = mae_loss(decoded[:, 1:], target) + sam_loss(decoded[:, 1:], target)
        return loss

    def train_epoch(self, dataloader: Iterable[Mapping[str, torch.Tensor]]) -> float:
        total = 0.0
        steps = 0
        self.model.train()
        for batch in dataloader:
            loss = self.training_step(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.ema is not None:
                self.ema.update(self.model)
            total += float(loss.detach())
            steps += 1
        return total / max(steps, 1)


__all__ = ["StageAExperiment", "StageAConfig"]
