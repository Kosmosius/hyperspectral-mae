from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True, cudnn_benchmark: bool = False) -> None:
    """
    Set all RNGs for reproducibility. Optionally force deterministic CuDNN.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = cudnn_benchmark
    if deterministic:
        torch.backends.cudnn.deterministic = True
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed for full determinism on some GPUs


def dataloader_worker_seed(worker_id: int) -> None:
    """
    Worker init fn for DataLoader. Ensures each worker has a proper RNG state.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
