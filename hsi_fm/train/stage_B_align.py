"""Lab-to-sensor alignment stage."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from hsi_fm.model.backbone import HyperspectralMAE
from hsi_fm.model.losses import cycle_loss, info_nce, masked_mse, sam_loss
from hsi_fm.physics.renderer import LabToSensorRenderer


@dataclass
class StageBLossConfig:
    recon: float = 1.0
    nce: float = 1.0
    cycle: float = 1.0


class StageBExperiment:
    """Alignment stage operating on lab and overhead spectra."""

    def __init__(
        self,
        model: HyperspectralMAE,
        renderer: LabToSensorRenderer,
        *,
        lab_dim: int,
        sensor_dim: int,
        proj_dim: int,
        device: torch.device,
        loss_cfg: StageBLossConfig,
        temperature: float = 0.1,
    ) -> None:
        self.model = model.to(device)
        self.renderer = renderer
        self.device = device
        self.loss_cfg = loss_cfg
        self.temperature = temperature
        self.lab_in = nn.Linear(lab_dim, model.embed_dim).to(device)
        self.over_in = nn.Linear(sensor_dim, model.embed_dim).to(device)
        self.lab_out = nn.Linear(model.embed_dim, lab_dim).to(device)
        self.over_out = nn.Linear(model.embed_dim, sensor_dim).to(device)
        self.proj_lab = nn.Sequential(
            nn.Linear(model.embed_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        ).to(device)
        self.proj_over = nn.Sequential(
            nn.Linear(model.embed_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        ).to(device)
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def parameters(self):  # pragma: no cover - simple wrapper
        yield from self.model.parameters()
        yield from self.lab_in.parameters()
        yield from self.over_in.parameters()
        yield from self.lab_out.parameters()
        yield from self.over_out.parameters()
        yield from self.proj_lab.parameters()
        yield from self.proj_over.parameters()

    def set_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer

    def _encode_tokens(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        decoded = self.model(tokens)
        cls = decoded[:, 0]
        recon = decoded[:, 1:]
        return cls, recon

    def _encode_lab(self, spectra: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.lab_in(spectra).unsqueeze(1)
        cls, recon = self._encode_tokens(tokens)
        recon = self.lab_out(recon).squeeze(1)
        return cls, recon

    def _encode_overhead(self, measurements: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.over_in(measurements).unsqueeze(1)
        cls, recon = self._encode_tokens(tokens)
        recon = self.over_out(recon).squeeze(1)
        return cls, recon

    def _inverse_batch(
        self,
        sensor: torch.Tensor,
        centers: torch.Tensor,
        fwhm: torch.Tensor,
        rt_params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        outputs = []
        for i in range(sensor.shape[0]):
            overrides = {key: float(rt_params[key][i].item()) for key in rt_params}
            refl, _ = self.renderer.inverse(
                sensor[i : i + 1],
                sensor_centers_nm=centers[i],
                sensor_fwhm_nm=fwhm[i],
                rt_overrides=overrides,
            )
            outputs.append(refl)
        return torch.cat(outputs, dim=0)

    def _render_batch(
        self,
        reflectance: torch.Tensor,
        centers: torch.Tensor,
        fwhm: torch.Tensor,
        rt_params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        outputs = []
        for i in range(reflectance.shape[0]):
            overrides = {key: float(rt_params[key][i].item()) for key in rt_params}
            sensor, _ = self.renderer.render(
                reflectance[i : i + 1],
                sensor_centers_nm=centers[i],
                sensor_fwhm_nm=fwhm[i],
                rt_overrides=overrides,
            )
            outputs.append(sensor)
        return torch.cat(outputs, dim=0)

    def _match_indices(
        self, lab_idx: torch.Tensor, over_idx: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        mapping = {int(val.item()): i for i, val in enumerate(lab_idx)}
        lab_positions = []
        over_positions = []
        for j, val in enumerate(over_idx):
            match = mapping.get(int(val.item()))
            if match is not None:
                lab_positions.append(match)
                over_positions.append(j)
        if lab_positions:
            lab_tensor = torch.tensor(lab_positions, device=self.device, dtype=torch.long)
            over_tensor = torch.tensor(over_positions, device=self.device, dtype=torch.long)
            return lab_tensor, over_tensor
        return None, None

    def training_step(
        self,
        lab_batch: Mapping[str, torch.Tensor],
        overhead_batch: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if self.optimizer is None:
            raise RuntimeError("Optimizer has not been set")

        lab_ref = lab_batch["reflectance_hi"].to(self.device)
        sensor = overhead_batch["radiance_sensor"].to(self.device)
        centers = overhead_batch["sensor_centers_nm"].to(self.device)
        fwhm = overhead_batch["sensor_fwhm_nm"].to(self.device)
        rt_params = {k: v.to(self.device) for k, v in overhead_batch["rt_params"].items()}

        lab_cls, lab_recon = self._encode_lab(lab_ref)
        over_cls, over_recon = self._encode_overhead(sensor)

        recon_lab = masked_mse(lab_recon, lab_ref, None) + sam_loss(lab_recon, lab_ref)
        recon_over = masked_mse(over_recon, sensor, None)
        recon_loss = 0.5 * (recon_lab + recon_over)

        lab_idx = lab_batch["index"].to(self.device)
        over_idx = overhead_batch["lab_index"].to(self.device)
        lab_pos_idx, over_pos_idx = self._match_indices(lab_idx, over_idx)
        if lab_pos_idx is not None and over_pos_idx is not None:
            z_lab = self.proj_lab(lab_cls)[lab_pos_idx]
            z_over = self.proj_over(over_cls)[over_pos_idx]
            loss_nce = info_nce(z_lab, z_over, self.temperature)
            cos_matrix = F.normalize(z_over, dim=-1) @ F.normalize(z_lab, dim=-1).T
            pos_cos = torch.diagonal(cos_matrix, 0).mean().item()
            neg_mask = torch.ones_like(cos_matrix, dtype=torch.bool)
            neg_mask[torch.arange(len(over_pos_idx)), torch.arange(len(lab_pos_idx))] = False
            neg_values = cos_matrix[neg_mask]
            neg_cos = float(neg_values.mean().item()) if neg_values.numel() > 0 else 0.0
        else:
            loss_nce = torch.zeros(1, device=self.device, dtype=lab_ref.dtype)
            pos_cos = 0.0
            neg_cos = 0.0

        lab_render, _ = self.renderer.render(lab_ref)
        lab_cycle, _ = self.renderer.inverse(lab_render)
        cycle_lab = cycle_loss(lab_ref, lab_cycle, None)

        over_ref = self._inverse_batch(sensor, centers, fwhm, rt_params)
        over_cycle = self._render_batch(over_ref, centers, fwhm, rt_params)
        cycle_over = masked_mse(over_cycle, sensor, None)
        cycle_total = 0.5 * (cycle_lab + cycle_over)

        total_loss = (
            self.loss_cfg.recon * recon_loss
            + self.loss_cfg.nce * loss_nce
            + self.loss_cfg.cycle * cycle_total
        )
        metrics = {
            "recon": float(recon_loss.detach().cpu()),
            "nce": float(loss_nce.detach().cpu()),
            "cycle": float(cycle_total.detach().cpu()),
            "pos_cos": pos_cos,
            "neg_cos": neg_cos,
        }
        return total_loss, metrics

    def train_epoch(
        self,
        lab_loader: Iterable[Mapping[str, torch.Tensor]],
        overhead_loader: Iterable[Mapping[str, torch.Tensor]],
        *,
        max_steps: Optional[int] = None,
        autocast_dtype: Optional[torch.dtype] = None,
    ) -> float:
        if self.optimizer is None:
            raise RuntimeError("Optimizer has not been set")

        self.model.train()
        self.lab_in.train()
        self.over_in.train()
        self.lab_out.train()
        self.over_out.train()
        self.proj_lab.train()
        self.proj_over.train()

        total = 0.0
        steps = 0
        pos_cos_vals: List[float] = []
        neg_cos_vals: List[float] = []
        cycle_vals: List[float] = []
        iterator = zip(lab_loader, overhead_loader)
        for step, (lab_batch, over_batch) in enumerate(iterator):
            if max_steps is not None and step >= max_steps:
                break
            context = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if autocast_dtype is not None and self.device.type == "cuda"
                else nullcontext()
            )
            with context:
                loss, metrics = self.training_step(lab_batch, over_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += float(loss.detach())
            steps += 1
            pos_cos_vals.append(metrics["pos_cos"])
            neg_cos_vals.append(metrics["neg_cos"])
            cycle_vals.append(metrics["cycle"])
        if steps > 0:
            pos_mean = sum(pos_cos_vals) / len(pos_cos_vals)
            neg_mean = sum(neg_cos_vals) / len(neg_cos_vals)
            cycle_mean = sum(cycle_vals) / len(cycle_vals)
            print(
                f"[stage_b] pos_cos={pos_mean:.3f} neg_cos={neg_mean:.3f} "
                f"cycle={cycle_mean:.4f}"
            )
        return total / max(steps, 1)


__all__ = ["StageBExperiment", "StageBLossConfig"]
