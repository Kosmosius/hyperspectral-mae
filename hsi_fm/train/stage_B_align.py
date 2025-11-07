"""Lab-to-sensor alignment stage."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from hsi_fm.model.losses import mae_loss, sam_loss
from typing import Dict

from hsi_fm.physics.renderer import LabToSensorRenderer
from hsi_fm.physics.srf import convolve_srf_torch


@dataclass
class StageBConfig:
    contrastive_weight: float = 1.0
    cycle_weight: float = 1.0


class StageBExperiment:
    def __init__(self, renderer: LabToSensorRenderer, encoder: torch.nn.Module):
        self.renderer = renderer
        self.encoder = encoder

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "lab_reflectance" in batch:
            reflectance = batch["lab_reflectance"]
        else:
            reflectance = batch["tokens"].mean(dim=1)
        wavelengths = batch.get("wavelengths")
        srf = batch.get("srf")
        srf_mask = batch.get("srf_mask")
        atmosphere = batch.get("atmosphere")

        if wavelengths is None:
            wavelengths = torch.linspace(400.0, 2500.0, reflectance.shape[-1], device=reflectance.device)

        if wavelengths.ndim == 1:
            wavelengths_list = wavelengths.unsqueeze(0).expand(reflectance.shape[0], -1)
        else:
            wavelengths_list = wavelengths

        if srf is not None:
            sensor_space = []
            for idx, (spec, wl, response) in enumerate(zip(reflectance, wavelengths_list, srf)):
                if srf_mask is not None:
                    valid_rows = srf_mask[idx].any(dim=-1)
                    valid_cols = srf_mask[idx][0]
                    response = response[valid_rows][:, valid_cols]
                    wl = wl[valid_cols]
                sensor_space.append(convolve_srf_torch(spec.unsqueeze(0), wl, response).squeeze(0))
            rendered_tensor = torch.stack(sensor_space)
        else:
            rendered_tensor = reflectance

        rendered = self.renderer.render(rendered_tensor, wavelengths_list[0], atmosphere)
        reconstructed = self.renderer.invert(rendered, wavelengths_list[0], atmosphere)
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
