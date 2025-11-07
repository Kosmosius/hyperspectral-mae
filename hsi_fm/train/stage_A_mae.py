"""Self-supervised pretraining stage."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Optional

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

    def __init__(
        self,
        model: HyperspectralMAE,
        *,
        token_dim: int,
        device: torch.device,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.input_proj = nn.Linear(token_dim, model.embed_dim).to(device)
        self.output_proj = nn.Linear(model.embed_dim, token_dim).to(device)
        self.optimizer = optimizer

    def parameters(self) -> Iterator[nn.Parameter]:
        for module in (self.input_proj, self.model, self.output_proj):
            yield from module.parameters()

    def set_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer

    def training_step(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.optimizer is None:
            raise RuntimeError("Optimizer has not been set")
        tokens = batch["tokens"].to(self.device)
        targets = batch["target"].to(self.device)
        mask = batch.get("tokens_mask")
        tokens_emb = self.input_proj(tokens)
        decoded = self.model(tokens_emb)
        recon = self.output_proj(decoded[:, 1:])
        if mask is not None:
            valid = mask.to(self.device)
            recon = recon[valid]
            targets = targets[valid]
        else:
            recon = recon.reshape(-1, recon.shape[-1])
            targets = targets.reshape(-1, targets.shape[-1])
        loss = mae_loss(recon, targets) + sam_loss(recon, targets)
        return loss

    def train_epoch(
        self,
        dataloader: Iterable[Mapping[str, torch.Tensor]],
        *,
        max_steps: Optional[int] = None,
        autocast_dtype: Optional[torch.dtype] = None,
        print_shapes: int = 0,
    ) -> float:
        if self.optimizer is None:
            raise RuntimeError("Optimizer has not been set")
        total = 0.0
        steps = 0
        self.model.train()
        self.input_proj.train()
        self.output_proj.train()
        for step, batch in enumerate(dataloader):
            if max_steps is not None and step >= max_steps:
                break
            if step < print_shapes:
                print(
                    f"[stage_a] step={step} tokens={batch['tokens'].shape} "
                    f"target={batch['target'].shape}"
                )
            if autocast_dtype is not None and self.device.type != "cuda":
                autocast_dtype = None
            context = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if autocast_dtype is not None
                else nullcontext()
            )
            with context:
                loss = self.training_step(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += float(loss.detach())
            steps += 1
        return total / max(steps, 1)


__all__ = ["StageAExperiment", "StageAConfig"]
