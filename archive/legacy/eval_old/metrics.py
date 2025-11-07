from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

# ---------- Helpers & typing ----------

ArrayLike = Union[np.ndarray, Tensor]

BinName = Literal["narrow", "medium", "wide"]


def _to_torch(x: ArrayLike, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> Tensor:
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    else:
        t = x
    if dtype is not None:
        t = t.to(dtype)
    if device is not None:
        t = t.to(device)
    return t


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


# ---------- Core spectral metrics ----------

def spectral_angle(f_hat: ArrayLike, f_true: ArrayLike, eps: float = 1e-8, radians: bool = True) -> Tensor:
    """
    Spectral Angle Mapper (SAM) per-sample.

    Args:
        f_hat: [B,K] predicted canonical reflectance
        f_true: [B,K] target canonical reflectance
        eps: small number for numerical stability
        radians: if False, returns degrees

    Returns:
        [B] tensor of angles
    """
    fh = _to_torch(f_hat, dtype=torch.float32)
    ft = _to_torch(f_true, dtype=torch.float32)
    dot = (fh * ft).sum(dim=-1).clamp_min(eps)
    n1 = fh.norm(p=2, dim=-1).clamp_min(eps)
    n2 = ft.norm(p=2, dim=-1).clamp_min(eps)
    cos = (dot / (n1 * n2)).clamp(-1.0, 1.0)
    ang = torch.arccos(cos)
    if not radians:
        ang = ang * (180.0 / math.pi)
    return ang


def per_lambda_rmse(f_hat: ArrayLike, f_true: ArrayLike, mask: Optional[ArrayLike] = None) -> Tensor:
    """
    Per-wavelength RMSE across the batch.

    Args:
        f_hat: [B,K]
        f_true: [B,K]
        mask:  [K] boolean (optional) 1 indicates valid wavelength

    Returns:
        [K] RMSE per wavelength (0 where mask==0 if provided).
    """
    fh = _to_torch(f_hat, dtype=torch.float32)
    ft = _to_torch(f_true, dtype=torch.float32)
    err2 = (fh - ft) ** 2
    if mask is not None:
        m = _to_torch(mask, dtype=torch.float32)
        err2 = err2 * m[None, :]
        denom = m.sum().clamp_min(1.0)
    else:
        denom = torch.tensor(fh.shape[0], dtype=torch.float32, device=fh.device)
    rmse = torch.sqrt(err2.sum(dim=0) / denom)
    if mask is not None:
        rmse = rmse * _to_torch(mask, dtype=torch.float32)
    return rmse


def curvature_energy(f_hat: ArrayLike, lambda_weights: Optional[ArrayLike] = None) -> Tensor:
    """
    Discrete curvature energy (sum of squared 2nd differences) per sample.

    Args:
        f_hat: [B,K]
        lambda_weights: [K] optional weights (lower in known absorption windows)

    Returns:
        [B] curvature energies
    """
    fh = _to_torch(f_hat, dtype=torch.float32)
    B, K = fh.shape
    if K < 3:
        return torch.zeros(B, dtype=torch.float32, device=fh.device)
    d2 = fh[:, 2:] - 2 * fh[:, 1:-1] + fh[:, :-2]  # [B,K-2]
    if lambda_weights is not None:
        w = _to_torch(lambda_weights, dtype=torch.float32)
        if w.numel() == K:
            w2 = w[2:]  # weight center bin of each second-diff
            d2 = d2 * w2.unsqueeze(0)
        elif w.numel() == K - 2:
            d2 = d2 * w.unsqueeze(0)
        else:
            raise ValueError("lambda_weights must be length K or K-2")
    return (d2 ** 2).sum(dim=1)


def nonneg_violation_rate(f_hat: ArrayLike, lo: float = 0.0, hi: float = 1.0) -> Tensor:
    """
    Fraction of bins outside [lo, hi] per sample.

    Args:
        f_hat: [B,K]
    """
    fh = _to_torch(f_hat, dtype=torch.float32)
    bad = (fh < lo) | (fh > hi)
    return bad.float().mean(dim=1)


# ---------- Band-space metrics (need SRFs / R) ----------

def effective_width_nm(R: ArrayLike, nm_per_step: Optional[float] = None) -> Tensor:
    """
    Top-hat equivalent width per band: EW = 1 / sum_k r_{ik}^2.
    If nm_per_step is provided, multiply by that to convert to [nm], otherwise returns [steps].
    Supports [N,K] or [B,N,K].
    """
    Rt = _to_torch(R, dtype=torch.float64)
    if Rt.dim() == 2:
        ew = 1.0 / Rt.pow(2).sum(dim=-1).clamp_min(1e-12)  # [N]
    elif Rt.dim() == 3:
        ew = 1.0 / Rt.pow(2).sum(dim=-1).clamp_min(1e-12)  # [B,N]
    else:
        raise ValueError("R must be [N,K] or [B,N,K]")
    if nm_per_step is not None:
        ew = ew * nm_per_step
    return ew.to(torch.float32)


def width_bins(ew_nm: Tensor, thresholds_nm: Tuple[float, float] = (10.0, 40.0)) -> Dict[BinName, Tensor]:
    """
    Assign bands to width bins (per-sample if ew is [B,N], otherwise global [N]).

    Returns boolean masks dict for 'narrow'/'medium'/'wide' with same leading dims as ew_nm.
    """
    t1, t2 = thresholds_nm
    narrow = ew_nm < t1
    medium = (ew_nm >= t1) & (ew_nm <= t2)
    wide = ew_nm > t2
    return {"narrow": narrow, "medium": medium, "wide": wide}


def band_mae_binned(y_hat: ArrayLike, y: ArrayLike, R: ArrayLike, nm_per_step: Optional[float] = None,
                    thresholds_nm: Tuple[float, float] = (10.0, 40.0)) -> Dict[BinName, Tensor]:
    """
    Compute mean absolute error per width bin. Bins follow effective width thresholding.

    Args:
        y_hat: [B,N]
        y:     [B,N]
        R:     [N,K] or [B,N,K]
        nm_per_step: convert EW from steps to nanometers (e.g., canonical grid step)
        thresholds_nm: (10, 40) by default

    Returns:
        Dict of 'narrow'/'medium'/'wide' -> scalar [B] MAE for each sample in that bin (0 if bin empty)
    """
    yh = _to_torch(y_hat, dtype=torch.float32)
    yt = _to_torch(y, dtype=torch.float32)
    ew = effective_width_nm(R, nm_per_step=nm_per_step)  # [B,N] or [N]

    if ew.dim() == 1:
        ew = ew.unsqueeze(0).expand_as(yh)

    masks = width_bins(ew, thresholds_nm)
    out: Dict[BinName, Tensor] = {}
    for name, m in masks.items():
        # Avoid NaNs when a sample has no bands in the bin: use safe denom
        denom = m.sum(dim=1).clamp_min(1.0)  # [B]
        mae = (torch.abs(yh - yt) * m).sum(dim=1) / denom
        out[name] = mae
    return out


# ---------- Calibration (ECE & reliability) ----------

@dataclass
class ReliabilityCurve:
    bin_edges: np.ndarray         # [B+1]
    sigma_mean: np.ndarray        # [B]
    abs_resid_mean: np.ndarray    # [B]
    abs_resid_std: np.ndarray     # [B]
    expected_abs_gauss: np.ndarray  # [B] = sqrt(2/pi) * sigma_mean
    counts: np.ndarray            # [B]


def reliability_by_sigma(residual: ArrayLike, sigma: ArrayLike, n_bins: int = 12) -> ReliabilityCurve:
    """
    Bin by predicted sigma, compute empirical |residual| vs expected |N(0, sigma^2)| = sqrt(2/pi)*sigma.

    Args:
        residual: [*,N] residuals
        sigma:    [*,N] predicted standard deviations
        n_bins: number of bins over sigma

    Returns:
        ReliabilityCurve (NumPy arrays)
    """
    r = _to_numpy(residual).astype(np.float64).ravel()
    s = _to_numpy(sigma).astype(np.float64).ravel()
    finite = np.isfinite(r) & np.isfinite(s) & (s > 0)
    r, s = r[finite], s[finite]

    # Define bins over sigma (quantiles are robust across datasets)
    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(s, q)
    edges[0] = min(edges[0], s.min())
    edges[-1] = max(edges[-1], s.max())

    idx = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, n_bins - 1)

    sigma_mean = np.zeros(n_bins, dtype=np.float64)
    abs_mean = np.zeros(n_bins, dtype=np.float64)
    abs_std = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)

    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        s_b = s[mask]
        r_b = np.abs(r[mask])
        sigma_mean[b] = float(np.mean(s_b))
        abs_mean[b] = float(np.mean(r_b))
        abs_std[b] = float(np.std(r_b))
        counts[b] = int(mask.sum())

    expected_abs = np.sqrt(2.0 / np.pi) * sigma_mean
    return ReliabilityCurve(bin_edges=edges, sigma_mean=sigma_mean, abs_resid_mean=abs_mean,
                            abs_resid_std=abs_std, expected_abs_gauss=expected_abs, counts=counts)


def expected_calibration_error_reliability(curve: ReliabilityCurve, weight_by_count: bool = True) -> float:
    """
    ECE over |residual| vs expected |N(0,sigma)| curves.

    ECE = sum_b w_b * |abs_resid_mean_b - expected_abs_b| / sum_b w_b,
    with w_b = counts if weight_by_count else 1.
    """
    w = curve.counts.astype(np.float64) if weight_by_count else np.ones_like(curve.counts, dtype=np.float64)
    diff = np.abs(curve.abs_resid_mean - curve.expected_abs_gauss)
    denom = max(1.0, float(w.sum()))
    return float((w * diff).sum() / denom)


# ---------- Bootstrap CIs ----------

def bootstrap_ci(
    values: ArrayLike,
    reps: int = 1000,
    alpha: float = 0.05,
    stratify: Optional[ArrayLike] = None,
    agg: Literal["mean", "median"] = "median",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval (BCa not required for routine CI; simple percentile CI here).

    Args:
        values: [N] numeric array
        reps: bootstrap repetitions
        alpha: (e.g., 0.05 -> 95% CI)
        stratify: [N] labels for stratified resampling (keeps per-class proportion)
        agg: aggregate ('mean' or 'median')
        rng: optional NumPy RNG

    Returns:
        (center, lo, hi)
    """
    v = _to_numpy(values).astype(np.float64)
    assert v.ndim == 1, "values must be 1D"
    N = v.shape[0]
    if rng is None:
        rng = np.random.default_rng()

    def aggregate(x: np.ndarray) -> float:
        return float(np.median(x)) if agg == "median" else float(np.mean(x))

    center = aggregate(v)

    if stratify is None:
        idx = np.arange(N)
        boot = np.empty(reps, dtype=np.float64)
        for i in range(reps):
            s = rng.choice(idx, size=N, replace=True)
            boot[i] = aggregate(v[s])
    else:
        strat = _to_numpy(stratify)
        uniq = np.unique(strat)
        boot = np.empty(reps, dtype=np.float64)
        for i in range(reps):
            pieces = []
            for u in uniq:
                mask = strat == u
                idx = np.where(mask)[0]
                k = idx.size
                s = np.random.choice(idx, size=k, replace=True)
                pieces.append(v[s])
            boot[i] = aggregate(np.concatenate(pieces, axis=0))

    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1.0 - alpha / 2))
    return center, lo, hi


# ---------- Convenience: aggregate a batch of predictions ----------

@dataclass
class BandspaceEval:
    mae_overall: float
    mae_bins: Dict[BinName, float]
    ece: float


def evaluate_bandspace(
    y_hat: ArrayLike,
    y: ArrayLike,
    R: ArrayLike,
    nm_per_step: Optional[float] = None,
    thresholds_nm: Tuple[float, float] = (10.0, 40.0),
    residual_sigma: Optional[Tuple[ArrayLike, ArrayLike]] = None,
) -> BandspaceEval:
    """
    Handy wrapper to compute band-space MAE, width-binned MAEs, and ECE if residual/sigma provided.
    """
    yh = _to_torch(y_hat, dtype=torch.float32)
    yt = _to_torch(y, dtype=torch.float32)

    overall_mae = torch.mean(torch.abs(yh - yt)).item()

    bins = band_mae_binned(yh, yt, R, nm_per_step=nm_per_step, thresholds_nm=thresholds_nm)
    mae_bins = {k: float(v.mean().item()) for k, v in bins.items()}

    ece = float("nan")
    if residual_sigma is not None:
        resid, sigma = residual_sigma
        curve = reliability_by_sigma(resid, sigma, n_bins=12)
        ece = expected_calibration_error_reliability(curve, weight_by_count=True)

    return BandspaceEval(mae_overall=overall_mae, mae_bins=mae_bins, ece=ece)
