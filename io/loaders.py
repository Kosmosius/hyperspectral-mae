from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    DataLoader = object  # type: ignore
    TORCH_AVAILABLE = False


@dataclass
class RaggedBatch:
    """
    A generic ragged batch for spectral samples with variable band counts N_i or
    different SRF operators per sample.

    Attributes:
      items: list of per-sample dicts (pass-through metadata)
      stacked: optional stacked arrays when rectangular (e.g., canonical f[K])
      to_torch(): converts numeric arrays to torch tensors if PyTorch available.
    """
    items: List[Dict[str, Any]]
    stacked: Dict[str, np.ndarray] | None = None

    def to_torch(self, device: Optional[str] = None) -> "RaggedBatch":
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        dev = torch.device(device) if device else torch.device("cpu")

        def _to_t(v):
            if isinstance(v, np.ndarray) and v.dtype.kind in ("f", "i", "u", "b"):
                return torch.from_numpy(v).to(dev)
            return v

        items = [{k: _to_t(v) for k, v in it.items()} for it in self.items]
        stacked = {k: _to_t(v) for k, v in (self.stacked or {}).items()} if self.stacked else None
        return RaggedBatch(items=items, stacked=stacked)  # type: ignore[return-value]


def collate_samples(batch: Sequence[Dict[str, Any]], prefer_torch: bool = False) -> RaggedBatch:
    """
    Collate function for DataLoader (or manual batching). It is tolerant of:
     - variable-length band vectors y (different N per sample),
     - differing SRF matrices R (different sensors),
     - optional canonical items that can be stacked (e.g., f_lambda[K]).

    Strategy:
      - Keep per-sample dicts in a list (ragged safe).
      - Attempt to stack keys that are clearly rectangular across batch (same shape & dtype).
      - If prefer_torch=True and torch is available, convert stacked to tensors; leave ragged items as-is.
    """
    if not isinstance(batch, (list, tuple)) or len(batch) == 0:
        return RaggedBatch(items=[])

    # Keys that are expected to be ragged (leave unstacked): R (operator), mask (per-band), y (per-band), grid, extra
    ragged_keys = {"R", "mask", "y", "grid", "extra"}

    # Collect items
    items: List[Dict[str, Any]] = []
    for sample in batch:
        items.append(sample)

    # Try to stack non-ragged numeric arrays of consistent shape
    stacked: Dict[str, np.ndarray] = {}
    # Inspect keys common to all
    common_keys = set.intersection(*(set(s.keys()) for s in batch))
    for k in sorted(common_keys):
        if k in ragged_keys:
            continue
        # Only stack simple numeric arrays
        arrs: List[np.ndarray] = []
        ok = True
        for s in batch:
            v = s[k]
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.dtype.kind in ("f", "i", "u", "b"):
                arrs.append(v)
            else:
                ok = False
                break
        if ok and len(arrs) == len(batch):
            # Check same shape
            shapes = {a.shape for a in arrs}
            if len(shapes) == 1:
                stacked[k] = np.stack(arrs, axis=0)

    rb = RaggedBatch(items=items, stacked=stacked if stacked else None)
    if prefer_torch and TORCH_AVAILABLE and rb.stacked:
        # Convert ONLY stacked arrays to tensors; ragged items stay as lists-of-dicts
        rb = rb.to_torch()
    return rb


def get_dataloader(
    dataset: Any,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    prefer_torch: bool = True,
) -> Any:
    """
    Thin wrapper that returns a PyTorch DataLoader if torch is available.
    Otherwise, returns a simple Python generator over batches.

    The collate function is `collate_samples`, which emits RaggedBatch.
    """
    if TORCH_AVAILABLE:
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, collate_fn=lambda b: collate_samples(b, prefer_torch=prefer_torch),
            pin_memory=False, drop_last=False,
        )

    # Fallback: naive Python batching
    def _iter():
        idxs = np.arange(len(dataset))
        if shuffle:
            rng = np.random.default_rng(0)
            rng.shuffle(idxs)
        for i in range(0, len(dataset), batch_size):
            batch = [dataset[int(j)] for j in idxs[i:i+batch_size]]
            yield collate_samples(batch, prefer_torch=False)

    return _iter()
