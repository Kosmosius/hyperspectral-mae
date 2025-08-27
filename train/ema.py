from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import nn


@dataclass
class EMAConfig:
    decay: float = 0.999
    device: Optional[torch.device] = None
    pinning: bool = True  # keep ema on same device as model by default


class EMA:
    """
    Exponential Moving Average of model parameters & buffers.
    - Keeps a shadow copy.
    - update(): ema = decay*ema + (1-decay)*model
    - store()/restore(): swap weights during eval/export if needed.
    """
    def __init__(self, model: nn.Module, cfg: EMAConfig = EMAConfig()):
        self.decay = float(cfg.decay)
        self.shadow = {}
        self.backup = {}
        # Initialize shadow
        for name, p in model.state_dict().items():
            self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.state_dict().items():
            assert name in self.shadow, f"Param {name} missing in EMA shadow"
            self.shadow[name].mul_(self.decary).add_(param.detach(), alpha=(1.0 - self.decary))

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        self.backup = {n: p.detach().clone() for n, p in model.state_dict().items()}

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if not self.backup:
            return
        model.load_state_dict(self.backup, strict=True)
        self.backup = {}
