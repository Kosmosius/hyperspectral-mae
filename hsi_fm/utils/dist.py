"""Utilities for distributed training and reproducibility."""
from __future__ import annotations

import contextlib
import os
import random
from datetime import timedelta
from typing import Iterator, Optional

import numpy as np
import torch
import torch.distributed as dist


def init_distributed(backend: Optional[str] = None, timeout: Optional[float] = None) -> None:
    """Initialise :mod:`torch.distributed` if requested."""

    if dist.is_available() is False:
        raise RuntimeError("torch.distributed is not available in this build of PyTorch")
    if dist.is_initialized():
        return

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")

    init_kwargs = {"backend": backend, "rank": rank, "world_size": world_size}
    if timeout is not None:
        init_kwargs["timeout"] = timedelta(seconds=float(timeout))
    dist.init_process_group(**init_kwargs)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def is_main_process() -> bool:
    return get_rank() == 0


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def distributed_zero_first(local_rank: int) -> Iterator[None]:
    if is_distributed() and local_rank != 0:
        barrier()
    yield
    if is_distributed() and local_rank == 0:
        barrier()


__all__ = [
    "init_distributed",
    "is_distributed",
    "get_rank",
    "get_world_size",
    "barrier",
    "is_main_process",
    "seed_all",
    "distributed_zero_first",
]

