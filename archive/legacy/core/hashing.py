from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple
import numpy as np


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_ndarray(arr: np.ndarray) -> str:
    """
    Hash an ndarray content with dtype, shape, endianness for stability.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("sha256_ndarray expects a numpy.ndarray")
    payload = {
        "dtype": str(arr.dtype),
        "shape": arr.shape,
        "endianness": ">" if arr.dtype.byteorder == ">" else "<",
    }
    meta = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    h = hashlib.sha256()
    h.update(meta)
    # Use C-contiguous view to hash raw bytes deterministically
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def sha256_path(path: str | Path, chunk_size: int = 1 << 20) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def stable_meta_hash(items: Iterable[Tuple[str, Any]]) -> str:
    """
    Hash a small metadata mapping (key, value) pairs with canonical JSON.
    Values must be JSON-serializable or numpy arrays (converted to list).
    """
    normalized: Dict[str, Any] = {}
    for k, v in items:
        if isinstance(v, np.ndarray):
            normalized[k] = v.tolist()
        else:
            normalized[k] = v
    blob = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(blob.encode("utf-8"))
