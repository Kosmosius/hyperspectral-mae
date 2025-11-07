from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class RFFSpec:
    """
    Specification for a Random Fourier Feature (RFF) bank over a scalar x ∈ [0,1].

    strategy:
      - "logspace": deterministic log-spaced frequencies in [f_min, f_max].
      - "gaussian": ω ~ N(0, 2πσ^2). Use sigma to control bandwidth.
    """
    strategy: str = "logspace"               # "logspace" | "gaussian"
    num_features: int = 32                    # number of frequency bins (each yields 2 features: sin/cos)
    f_min: float = 2.0                        # min frequency (for logspace)
    f_max: float = 64.0                       # max frequency (for logspace)
    sigma: float = 10.0                       # std for gaussian ω (ignored for logspace)
    seed: int = 42                            # RNG for gaussian and phase shifts
    add_bias: bool = False                    # if True, prepend a learned/constant 1 feature (cos(0·x)=1)


class RFFBank:
    """
    Random Fourier Feature bank that maps x ∈ [0,1]^B to Φ(x) ∈ R^{2M (+1)}.

    - Deterministic when strategy="logspace".
    - Reproducible when strategy="gaussian" (seeded).
    - Supports persistence to .npz with metadata.

    Encoding: [sin(2π ω_k x + b_k), cos(2π ω_k x + b_k)] for k=1..M.
    Phase shifts b_k are randomized for mild decorrelation (seeded).
    """
    def __init__(self, spec: RFFSpec):
        self.spec = spec
        self._rng = np.random.default_rng(spec.seed)

        if spec.strategy not in {"logspace", "gaussian"}:
            raise ValueError("RFFSpec.strategy must be 'logspace' or 'gaussian'")

        if spec.num_features <= 0:
            raise ValueError("RFFSpec.num_features must be > 0")

        if spec.strategy == "logspace":
            # Log-spaced frequencies (excluding 0), deterministic.
            self.omegas = np.geomspace(spec.f_min, spec.f_max, spec.num_features, dtype=np.float64)
        else:
            # Gaussian frequencies centered at 0 (radial frequency domain), magnitude in cycles over [0,1].
            self.omegas = np.abs(self._rng.normal(loc=0.0, scale=spec.sigma, size=spec.num_features)).astype(np.float64)

        # Random phases for both strategies to reduce aliasing artifacts
        self.bias = self._rng.uniform(0.0, 2 * np.pi, size=spec.num_features).astype(np.float64)

    @property
    def out_dim(self) -> int:
        return (2 * self.spec.num_features) + (1 if self.spec.add_bias else 0)

    def encode(self, x01: np.ndarray) -> np.ndarray:
        """
        Encode normalized scalar(s) x ∈ [0,1] to RFF features.

        Args:
            x01: np.ndarray shape [B] or [B,1] or scalar; must lie in [0,1].

        Returns:
            features: np.ndarray of shape [B, D] where D = out_dim
        """
        x = np.asarray(x01, dtype=np.float64)
        if x.ndim == 0:
            x = x[None]
        elif x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        elif x.ndim != 1:
            raise ValueError("x must be 1D or [B,1]")

        if np.any(x < -1e-9) or np.any(x > 1 + 1e-9):
            raise ValueError("RFF encode expects x normalized to [0,1]")

        # Broadcast to [B, M]
        X = x[:, None]  # [B,1]
        O = self.omegas[None, :]  # [1,M]
        B = self.bias[None, :]    # [1,M]
        arg = 2 * np.pi * O * X + B

        sin = np.sin(arg)
        cos = np.cos(arg)
        feats = np.concatenate([sin, cos], axis=1)  # [B, 2M]

        if self.spec.add_bias:
            feats = np.concatenate([np.ones((feats.shape[0], 1), dtype=feats.dtype), feats], axis=1)
        return feats.astype(np.float32)

    # ---- Persistence ----

    def to_state(self) -> Dict[str, object]:
        return {
            "spec": self.spec.__dict__,
            "omegas": self.omegas,
            "bias": self.bias,
        }

    @classmethod
    def from_state(cls, state: Dict[str, object]) -> "RFFBank":
        spec = RFFSpec(**{k: state["spec"][k] for k in state["spec"]})  # type: ignore[index]
        obj = cls(spec)
        obj.omegas = np.asarray(state["omegas"], dtype=np.float64)
        obj.bias = np.asarray(state["bias"], dtype=np.float64)
        return obj


def save_rff_bank(bank: RFFBank, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    state = bank.to_state()
    # Save arrays in .npz and JSON metadata as a string for transparency
    np.savez_compressed(
        p,
        omegas=state["omegas"],
        bias=state["bias"],
        spec_json=json.dumps(state["spec"], sort_keys=True),
    )


def load_rff_bank(path: str | Path) -> RFFBank:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    with np.load(p, allow_pickle=True) as z:
        omegas = z["omegas"].astype(np.float64)
        bias = z["bias"].astype(np.float64)
        spec_json = json.loads(str(z["spec_json"]))
    spec = RFFSpec(**spec_json)
    bank = RFFBank(spec)
    bank.omegas = omegas
    bank.bias = bias
    return bank
