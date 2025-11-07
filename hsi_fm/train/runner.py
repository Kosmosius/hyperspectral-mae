"""Minimal training loop utilities."""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Protocol

import torch


class TrainBatch(Protocol):
    def __getitem__(self, key: str) -> torch.Tensor:  # pragma: no cover - protocol definition
        ...


class SupportsTrainingStep(Protocol):
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer

    def training_step(self, batch: TrainBatch) -> torch.Tensor:
        ...


@dataclass
class LoopConfig:
    """Configuration for the training loop."""

    max_steps: int


class TrainingLoop:
    """Light-weight helper to run a fixed number of optimization steps."""

    def __init__(
        self,
        experiment: SupportsTrainingStep,
        dataloader: Iterable[TrainBatch],
        config: LoopConfig,
    ) -> None:
        self.experiment = experiment
        self.dataloader = dataloader
        self.config = config

    def run(self) -> float:
        """Execute ``config.max_steps`` optimization steps."""

        iterator = itertools.islice(self.dataloader, self.config.max_steps)
        total = 0.0
        steps = 0
        model = self.experiment.model
        model.train()
        for batch in iterator:
            loss = self.experiment.training_step(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.experiment.optimizer.step()
            self.experiment.optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach())
            steps += 1
        return total / max(steps, 1)


__all__ = ["TrainingLoop", "LoopConfig"]
