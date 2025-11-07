"""Reproducibility helpers for training."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True, cudnn_benchmark: bool = False) -> None:
    """Seed all random number generators for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = cudnn_benchmark
    if deterministic:
        torch.backends.cudnn.deterministic = True
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def dataloader_worker_seed(worker_id: int) -> None:
    """Worker init hook for :class:`torch.utils.data.DataLoader`."""

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


__all__ = ["set_seed", "dataloader_worker_seed"]
