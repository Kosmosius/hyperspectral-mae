from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor

# We intentionally keep this wrapper light on imports so "core" remains NumPy/Scipy-only.
# These names should exist in your core renderer implementation.
try:
    from ..core.renderer import build_R  # preferred simple functional API
except Exception:  # pragma: no cover
    build_R = None

try:
    from ..core.hashing import sha256_array
except Exception:  # pragma: no cover
    sha256_array = None


def _hash_srfs(srfs: Sequence[Any]) -> str:
    """
    Build a stable hash for an iterable of SRF specs.
    Each SRF is expected to have either:
      - dict-like with 'wavelength' and 'response' arrays, or
      - numpy array weights already aligned to canonical grid, or
      - attributes `.wavelength`, `.response`
    """
    import hashlib
    h = hashlib.sha256()
    for s in srfs:
        if isinstance(s, dict) and "wavelength" in s and "response" in s:
            w = np.asarray(s["wavelength"])
            r = np.asarray(s["response"])
        elif hasattr(s, "wavelength") and hasattr(s, "response"):
            w = np.asarray(getattr(s, "wavelength"))
            r = np.asarray(getattr(s, "response"))
        elif isinstance(s, np.ndarray):
            w = None
            r = s
        else:
            raise TypeError("Unsupported SRF spec for hashing")
        if w is not None:
            h.update(np.ascontiguousarray(w).tobytes())
        h.update(np.ascontiguousarray(r).tobytes())
    return h.hexdigest()


@lru_cache(maxsize=256)
def _cached_R(hash_key: str) -> Tensor:
    """
    Placeholder cache slot. The actual population happens through build_operator_torch
    which computes R then stores it in this LRU via its hash key.
    """
    raise RuntimeError("Internal cache miss without population. Use build_operator_torch to populate.")


def build_operator_torch(
    grid: Any,                     # core.grid.CanonicalGrid or compatible
    srfs: Sequence[Any],           # iterable of SRF specs compatible with core.renderer.build_R
    quadrature: str = "gauss",
    normalize: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """
    Build the SRF operator R [N,K] as a torch tensor by delegating to core.renderer (NumPy),
    then converting to torch and caching by SRF hash.
    """
    if build_R is None:
        raise ImportError("core.renderer.build_R not found; ensure core/renderer.py exposes build_R(grid, srfs, ...).")

    R_np: np.ndarray = build_R(grid=grid, srfs=srfs, quadrature=quadrature, normalize=normalize)  # [N,K]
    if not isinstance(R_np, np.ndarray):
        raise TypeError("core.renderer.build_R must return a numpy.ndarray")

    key = _hash_srfs(srfs)
    R_t = torch.from_numpy(R_np.copy())
    if device is not None or dtype is not None:
        R_t = R_t.to(device=device or torch.device("cpu"), dtype=dtype or torch.get_default_dtype())

    # Populate LRU cache by calling the cached function in a try/except pattern.
    try:
        _cached_R.cache_clear()  # avoid cache explosion if hash collisions happen
        def _store(_: str) -> Tensor:
            return R_t
        _cached_R.__wrapped__ = _store  # type: ignore[attr-defined]
        return _cached_R(key)           # store and return
    finally:
        # Restore wrapper
        _cached_R.__wrapped__ = lambda k: (_ for _ in ()).throw(RuntimeError("Use build_operator_torch"))  # type: ignore


def render_bandspace(
    f_hat: Tensor,     # [B,K] or [K]
    R: Tensor,         # [N,K] or [B,N,K]
    c_hat: Tensor | None = None,
) -> Tensor:
    """
    Compute y_hat = R @ (f_hat * c_hat) with broadcasting across batch.
    """
    if f_hat.dim() == 1:
        f_hat = f_hat.unsqueeze(0)
    if c_hat is not None:
        g = f_hat * c_hat
    else:
        g = f_hat

    if R.dim() == 2:
        y = torch.einsum("nk,bk->bn", R, g)
    elif R.dim() == 3:
        y = torch.einsum("bnk,bk->bn", R, g)
    else:
        raise ValueError("R must be [N,K] or [B,N,K]")
    return y
