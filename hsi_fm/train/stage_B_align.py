"""Lab-to-sensor alignment stage."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from hsi_fm.model.losses import mae_loss, sam_loss
from hsi_fm.physics.renderer import LabToSensorRenderer


@dataclass
class StageBConfig:
    contrastive_weight: float = 1.0
    cycle_weight: float = 1.0


class StageBExperiment:
    def __init__(self, renderer: LabToSensorRenderer, encoder: torch.nn.Module):
        self.renderer = renderer
        self.encoder = encoder

    def training_step(self, batch: dict) -> torch.Tensor:
        reflectance = batch["lab_reflectance"]
        wavelengths = batch["wavelengths"]
        atmosphere = batch.get("atmosphere")
        rendered = self.renderer.render(reflectance, wavelengths, atmosphere)
        reconstructed = self.renderer.invert(rendered, wavelengths, atmosphere)
        return mae_loss(reconstructed, reflectance) + sam_loss(reconstructed, reflectance)

    def train_epoch(self, dataloader: Iterable[dict]) -> float:
        total = 0.0
        steps = 0
        for batch in dataloader:
            loss = self.training_step(batch)
            total += float(loss)
            steps += 1
        return total / max(steps, 1)


__all__ = ["StageBExperiment", "StageBConfig"]
