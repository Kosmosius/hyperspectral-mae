"""Learning rate schedulers."""
from __future__ import annotations

from dataclasses import dataclass

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


@dataclass
class CosineWarmupConfig:
    warmup_steps: int
    max_steps: int


def cosine_with_warmup(optimizer: Optimizer, config: CosineWarmupConfig) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return (step + 1) / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1 + (1 - progress * 2) ** 3)

    return LambdaLR(optimizer, lr_lambda)


__all__ = ["CosineWarmupConfig", "cosine_with_warmup"]
