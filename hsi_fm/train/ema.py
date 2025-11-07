"""Exponential moving average utilities for model parameters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn


@dataclass
class EMAConfig:
    """Configuration options for :class:`EMA`."""

    decay: float = 0.999
    device: Optional[torch.device] = None
    pinning: bool = True


class EMA:
    """Maintain an exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, cfg: EMAConfig | None = None):
        self.cfg = cfg or EMAConfig()
        self.decay = float(self.cfg.decay)
        self.device = self.cfg.device
        self.shadow: Dict[str, Tensor] = {}
        self.backup: Dict[str, Tensor] = {}
        self._build_shadow(model)

    def _to_shadow(self, tensor: Tensor) -> Tensor:
        if self.device is not None:
            return tensor.detach().clone().to(self.device)
        return tensor.detach().clone()

    def _build_shadow(self, model: nn.Module) -> None:
        for name, param in model.state_dict().items():
            self.shadow[name] = self._to_shadow(param)
        if self.cfg.pinning and self.device is None and self.shadow:
            self.device = next(iter(self.shadow.values())).device

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.state_dict().items():
            if name not in self.shadow:
                self.shadow[name] = self._to_shadow(param)
                continue
            target = param.detach().clone()
            if self.device is not None:
                target = target.to(self.device)
            self.shadow[name].mul_(self.decay).add_(target, alpha=1.0 - self.decay)

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        self.backup = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        if not self.shadow:
            return
        model.load_state_dict({name: tensor.to(model.state_dict()[name].device) for name, tensor in self.shadow.items()})

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if not self.backup:
            return
        model.load_state_dict(self.backup)
        self.backup = {}


__all__ = ["EMA", "EMAConfig"]
