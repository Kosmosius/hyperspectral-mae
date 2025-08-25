from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ..core.types import NDArrayF
from ..core.grid import CanonicalGrid


# ---------- helpers ----------

def _normalize_to_01(x: np.ndarray, lo: float, hi: float, eps: float = 1e-12) -> np.ndarray:
    if hi <= lo:
        raise ValueError("Normalization bounds must satisfy hi > lo")
    return np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0)


def compute_centers_widths_from_R(R: np.ndarray, grid: CanonicalGrid) -> Tuple[np.ndarray, np.ndarray]:
    """
    Derive effective band centers and widths (in nm) from an SRF row-stochastic matrix R[N,K]
    on the canonical grid.

    - Center (nm): weighted centroid: sum_k r_{i,k} * λ_k
    - Width (nm):  Effective width (EW) in nm ~ (1 / sum_k r_{i,k}^2) * Δλ
                   (top-hat intuition; Δλ is grid step in nm)

    Args:
        R: [N, K] nonnegative, rows sum to 1 within numerical tolerance.
        grid: CanonicalGrid defining λ_k and Δλ.

    Returns:
        (centers_nm[N], widths_nm[N])
    """
    if R.ndim != 2:
        raise ValueError("R must be [N,K]")
    N, K = R.shape
    if K != grid.K:
        raise ValueError("R columns must match CanonicalGrid length")

    lambdas = grid.lambdas_nm.astype(np.float64)[None, :]  # [1,K]
    row_sums = R.sum(axis=1, keepdims=True)  # [N,1]
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        # Normalize defensively
        row_sums = np.where(row_sums <= 0.0, 1.0, row_sums)
        R = R / row_sums

    centers_nm = (R * lambdas).sum(axis=1)  # [N]

    ew_bins = 1.0 / (np.square(R).sum(axis=1) + 1e-12)  # bins
    widths_nm = ew_bins * grid.delta_nm  # nm
    return centers_nm.astype(np.float64), widths_nm.astype(np.float64)


# ---------- config & outputs ----------

@dataclass(frozen=True)
class TokenizerConfig:
    """
    Configuration for USR tokenization.

    Attributes:
        rff_c_dim:            number of frequency bins for center c (each -> 2 features)
        rff_delta_dim:        number of frequency bins for width Δ (each -> 2 features)
        rff_c_fmin/fmax:      frequency range for log-spaced center encoding
        rff_delta_fmin/fmax:  frequency range for log-spaced width encoding
        wavelength_min_nm/max_nm: bounds to normalize centers c -> [0,1]
        width_min_nm/max_nm:  bounds to normalize widths Δ -> [0,1]
        add_bias:             include a leading bias feature in each RFF
        s_dim:                expected dimension of SRF projection s (set 0 to disable)
        drop_s_prob:          probability to drop s at tokenization (simulates missing SRF shapes)
        jitter_c_nm:          Gaussian jitter (std, nm) for center c (data augmentation)
        jitter_delta_nm:      Gaussian jitter (std, nm) for width Δ (data augmentation)
        clip_y:               clip y to [-0.01, 1.02] (numeric stability)
        dtype:                float32 recommended for tokens
        sort_by_center:       sort tokens by center wavelength before returning
    """
    rff_c_dim: int = 32
    rff_delta_dim: int = 8
    rff_c_fmin: float = 2.0
    rff_c_fmax: float = 64.0
    rff_delta_fmin: float = 2.0
    rff_delta_fmax: float = 32.0
    wavelength_min_nm: float = 400.0
    wavelength_max_nm: float = 2500.0
    width_min_nm: float = 5.0
    width_max_nm: float = 80.0
    add_bias: bool = False
    s_dim: int = 16
    drop_s_prob: float = 0.0
    jitter_c_nm: float = 0.0
    jitter_delta_nm: float = 0.0
    clip_y: bool = True
    dtype: str = "float32"
    sort_by_center: bool = True
    seed: int = 1234


@dataclass(frozen=True)
class TokenizeResult:
    tokens: np.ndarray           # [N, d_t]
    fields: Dict[str, slice]     # keys: 'y', 'phi_c', 'phi_delta', optionally 's'
    y: np.ndarray                # [N]
    centers_nm: np.ndarray       # [N]
    widths_nm: np.ndarray        # [N]
    c01: np.ndarray              # [N] normalized centers
    d01: np.ndarray              # [N] normalized widths
    s: Optional[np.ndarray]      # [N, s_dim] or None
    order: np.ndarray            # [N] original indices after sorting (if any)


# ---------- tokenizer ----------

class Tokenizer:
    """
    Universal Spectral Representation (USR) tokenizer.

    Construct per-band tokens:
        t_i = [ y_i | ϕ_c(c_i) | ϕ_Δ(Δ_i) | s_i ]

    - ϕ_c and ϕ_Δ are log-spaced RFFs over normalized [0,1] inputs.
    - c_i, Δ_i can be provided directly OR derived from R rows and CanonicalGrid.
    - s_i is optional SRF shape projection on a provided orthonormal basis B[K, M_s].
      If missing, s_i can be dropped with probability `drop_s_prob` or zeroed.

    This module is NumPy-only for portability.
    """
    def __init__(self, config: TokenizerConfig):
        self.cfg = config
        # Initialize internal RFF banks (deterministic logspace)
        from .rff import RFFBank, RFFSpec  # local import to keep dependency surface minimal

        self._rng = np.random.default_rng(self.cfg.seed)

        self.rff_c = RFFBank(RFFSpec(
            strategy="logspace",
            num_features=self.cfg.rff_c_dim,
            f_min=self.cfg.rff_c_fmin,
            f_max=self.cfg.rff_c_fmax,
            seed=self.cfg.seed + 1,
            add_bias=self.cfg.add_bias,
        ))
        self.rff_delta = RFFBank(RFFSpec(
            strategy="logspace",
            num_features=self.cfg.rff_delta_dim,
            f_min=self.cfg.rff_delta_fmin,
            f_max=self.cfg.rff_delta_fmax,
            seed=self.cfg.seed + 2,
            add_bias=self.cfg.add_bias,
        ))

    @property
    def phi_c_dim(self) -> int:
        return self.rff_c.out_dim

    @property
    def phi_delta_dim(self) -> int:
        return self.rff_delta.out_dim

    def _maybe_jitter(self, x_nm: np.ndarray, std_nm: float) -> np.ndarray:
        if std_nm <= 0:
            return x_nm
        return x_nm + self._rng.normal(0.0, std_nm, size=x_nm.shape).astype(np.float64)

    def _build_s(
        self,
        R: Optional[np.ndarray],
        basis_K_M: Optional[np.ndarray],
        provided_s: Optional[np.ndarray],
        drop_prob: float,
    ) -> Optional[np.ndarray]:
        """
        Compute or vet SRF projection s.

        If provided_s is given, use it (with optional dropout).
        Else if R and basis_K_M provided, compute s_i = R_i · basis_K_M  (since rows of R sum to 1).
        Else return None.
        """
        s = None
        if provided_s is not None:
            s = np.asarray(provided_s, dtype=np.float64)
        elif (R is not None) and (basis_K_M is not None):
            # basis_K_M: [K, M_s]; R: [N, K]
            if basis_K_M.ndim != 2:
                raise ValueError("basis_K_M must be [K, M_s]")
            if R.shape[1] != basis_K_M.shape[0]:
                raise ValueError("R columns must match basis rows (K)")
            s = R @ basis_K_M  # [N, M_s]
        else:
            return None

        # Validate s dimension against config.s_dim if non-zero
        if self.cfg.s_dim and s.shape[1] != self.cfg.s_dim:
            raise ValueError(f"s dimension mismatch: got {s.shape[1]}, expected {self.cfg.s_dim}")

        # Apply dropout: set entire s_i to zeros with probability drop_prob
        if drop_prob > 0:
            keep_mask = self._rng.uniform(0.0, 1.0, size=(s.shape[0],)) >= drop_prob
            s = s.copy()
            s[~keep_mask, :] = 0.0

        return s.astype(np.float32)

    def tokenize(
        self,
        y: NDArrayF,                                 # [N] band values (reflectance/radiance; clipping optional)
        centers_nm: Optional[NDArrayF] = None,       # [N] band centers; if None, must pass R+grid
        widths_nm: Optional[NDArrayF] = None,        # [N] band widths; if None, must pass R+grid
        R: Optional[NDArrayF] = None,                # [N,K] SRF matrix (rowsum=1)
        grid: Optional[CanonicalGrid] = None,        # required if centers/widths are derived from R
        srf_basis_K_M: Optional[NDArrayF] = None,    # [K, M_s] orthonormal basis evaluated on canonical grid
        s_proj: Optional[NDArrayF] = None,           # [N, M_s] optional precomputed s
    ) -> TokenizeResult:
        """
        Build USR tokens per band.

        Returns TokenizeResult with:
          - tokens [N, d_t]
          - fields slices for 'y', 'phi_c', 'phi_delta', optionally 's'
          - diagnostic arrays (centers_nm, widths_nm, normalized c01/d01, s, order)
        """
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        N = y.shape[0]
        if self.cfg.clip_y:
            y = np.clip(y, -0.01, 1.02)

        # Derive c, Δ if needed
        if centers_nm is None or widths_nm is None:
            if R is None or grid is None:
                raise ValueError("If centers/widths are not provided, R and grid must be provided")
            centers_nm_d, widths_nm_d = compute_centers_widths_from_R(np.asarray(R, dtype=np.float64), grid)
            if centers_nm is None:
                centers_nm = centers_nm_d
            if widths_nm is None:
                widths_nm = widths_nm_d

        centers_nm = np.asarray(centers_nm, dtype=np.float64).reshape(-1)
        widths_nm = np.asarray(widths_nm, dtype=np.float64).reshape(-1)

        if centers_nm.shape[0] != N or widths_nm.shape[0] != N:
            raise ValueError("y, centers_nm, widths_nm must have same length")

        # Augment (jitter)
        if self.cfg.jitter_c_nm > 0:
            centers_nm = self._maybe_jitter(centers_nm, self.cfg.jitter_c_nm)
        if self.cfg.jitter_delta_nm > 0:
            widths_nm = np.maximum(1e-3, self._maybe_jitter(widths_nm, self.cfg.jitter_delta_nm))

        # Normalize to [0,1]
        c01 = _normalize_to_01(centers_nm, self.cfg.wavelength_min_nm, self.cfg.wavelength_max_nm)
        d01 = _normalize_to_01(widths_nm, self.cfg.width_min_nm, self.cfg.width_max_nm)

        # RFF encodings
        phi_c = self.rff_c.encode(c01)          # [N, Dc]
        phi_d = self.rff_delta.encode(d01)      # [N, Dd]

        # s projection (optional)
        s = self._build_s(
            R=np.asarray(R) if R is not None else None,
            basis_K_M=np.asarray(srf_basis_K_M) if srf_basis_K_M is not None else None,
            provided_s=np.asarray(s_proj) if s_proj is not None else None,
            drop_prob=self.cfg.drop_s_prob,
        )

        # Build token matrix
        dtype = np.float32 if self.cfg.dtype == "float32" else np.float64
        parts = []
        fields: Dict[str, slice] = {}

        start = 0
        y_col = y.astype(dtype).reshape(-1, 1)
        parts.append(y_col)
        fields["y"] = slice(start, start + 1)
        start += 1

        parts.append(phi_c.astype(dtype))
        fields["phi_c"] = slic_]()
