"""Task head fine-tuning stage."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from hsi_fm.model.heads_gas import GasPlumeHead
from hsi_fm.model.heads_solids import SolidsUnmixingHead
from hsi_fm.model.losses import dice_loss, mae_loss


@dataclass
class StageCConfig:
    solids_weight: float = 1.0
    gases_weight: float = 1.0


class StageCExperiment:
    def __init__(
        self,
        solids_head: SolidsUnmixingHead,
        gas_head: GasPlumeHead,
        optimizer: torch.optim.Optimizer,
    ):
        self.solids_head = solids_head
        self.gas_head = gas_head
        self.optimizer = optimizer

    def training_step(self, batch: dict) -> torch.Tensor:
        solids_features = batch["solids_features"]
        gas_features = batch["gas_features"]
        abundance_target = batch["abundance_target"]
        gas_mask = batch["gas_mask"]
        gas_column = batch["gas_column"]

        solids_output = self.solids_head(solids_features)
        num_endmembers = abundance_target.shape[-1]
        abundances = solids_output[..., :num_endmembers]
        recon = solids_output[..., num_endmembers:]
        solids_loss = mae_loss(abundances, abundance_target) + mae_loss(recon, batch["solids_recon_target"])

        gas_output = self.gas_head(gas_features)
        num_gases = gas_mask.shape[1]
        seg = gas_output[:, :num_gases]
        column = gas_output[:, num_gases:]
        gas_loss = dice_loss(seg, gas_mask) + mae_loss(column, gas_column)

        loss = solids_loss + gas_loss
        return loss

    def train_epoch(self, dataloader: Iterable[dict]) -> float:
        total = 0.0
        steps = 0
        for batch in dataloader:
            loss = self.training_step(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += float(loss.detach())
            steps += 1
        return total / max(steps, 1)


__all__ = ["StageCExperiment", "StageCConfig"]
