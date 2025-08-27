from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .metrics import (
    BandspaceEval,
    ReliabilityCurve,
    expected_calibration_error_reliability,
    nonneg_violation_rate,
    per_lambda_rmse,
    reliability_by_sigma,
    spectral_angle,
    curvature_energy,
    evaluate_bandspace,
)

# NOTE: Do not set global styles or colors to keep plots environment-agnostic per project policies.


# ---------- Data containers ----------

@dataclass
class CanonicalReport:
    sam_rad_median: float
    sam_rad_ci: Tuple[float, float]
    cr_mse_median: float
    cr_mse_ci: Tuple[float, float]
    curvature_median: float
    curvature_ci: Tuple[float, float]
    nvr_median: float
    nvr_ci: Tuple[float, float]


@dataclass
class BandspaceReport:
    mae_overall: float
    mae_bins: Dict[str, float]
    ece: float


@dataclass
class EvalReport:
    canonical: Optional[CanonicalReport]
    bandspace: Optional[BandspaceReport]
    per_lambda_rmse: Optional[np.ndarray]
    wavelength_nm: Optional[np.ndarray]
    reliability: Optional[ReliabilityCurve]


# ---------- Plots ----------

def plot_per_lambda_rmse(wavelength_nm: np.ndarray, rmse: np.ndarray, mask: Optional[np.ndarray] = None,
                         title: str = "Per-wavelength RMSE", out_path: Optional[str] = None) -> None:
    plt.figure()
    x = wavelength_nm
    y = rmse
    plt.plot(x, y, linewidth=1.5)
    if mask is not None:
        # Shade masked regions
        m = mask.astype(bool)
        # mark by zeroing? We'll draw a step overlay for visibility
        plt.plot(x[m], y[m], linewidth=1.5)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("RMSE")
    plt.title(title)
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150)
    plt.close()


def plot_width_binned_mae(mae_bins: Dict[str, float], title: str = "Width-binned Band MAE", out_path: Optional[str] = None) -> None:
    plt.figure()
    names = ["narrow", "medium", "wide"]
    vals = [mae_bins.get(n, np.nan) for n in names]
    plt.bar(range(len(names)), vals)
    plt.xticks(range(len(names)), names)
    plt.ylabel("MAE")
    plt.title(title)
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150)
    plt.close()


def plot_reliability(curve: ReliabilityCurve, title: str = "Calibration: |residual| vs expected", out_path: Optional[str] = None) -> None:
    plt.figure()
    # X-axis: predicted expected |residual| = sqrt(2/pi)*sigma_mean
    x = curve.expected_abs_gauss
    y = curve.abs_resid_mean
    plt.scatter(x, y, s=np.clip(curve.counts, 5, 80))  # point size ~ count
    # reference line y=x
    minv = float(min(np.nanmin(x), np.nanmin(y)))
    maxv = float(max(np.nanmax(x), np.nanmax(y)))
    plt.plot([minv, maxv], [minv, maxv])
    plt.xlabel("Expected |residual| (Gaussian)")
    plt.ylabel("Empirical |residual|")
    plt.title(title)
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150)
    plt.close()


# ---------- Tables & serialization ----------

def dataframe_metrics_summary(canon: Optional[CanonicalReport], band: Optional[BandspaceReport]) -> pd.DataFrame:
    rows = []
    if canon is not None:
        rows.extend([
            {"metric": "SAM (rad)", "value": canon.sam_rad_median, "ci_low": canon.sam_rad_ci[0], "ci_high": canon.sam_rad_ci[1]},
            {"metric": "CR-MSE", "value": canon.cr_mse_median, "ci_low": canon.cr_mse_ci[0], "ci_high": canon.cr_mse_ci[1]},
            {"metric": "Curvature", "value": canon.curvature_median, "ci_low": canon.curvature_ci[0], "ci_high": canon.curvature_ci[1]},
            {"metric": "NVR", "value": canon.nvr_median, "ci_low": canon.nvr_ci[0], "ci_high": canon.nvr_ci[1]},
        ])
    if band is not None:
        rows.append({"metric": "Band MAE (overall)", "value": band.mae_overall, "ci_low": None, "ci_high": None})
        for k, v in band.mae_bins.items():
            rows.append({"metric": f"Band MAE ({k})", "value": v, "ci_low": None, "ci_high": None})
        rows.append({"metric": "Calibration ECE", "value": band.ece, "ci_low": None, "ci_high": None})
    df = pd.DataFrame(rows)
    return df


def save_report_artifacts(
    out_dir: str,
    report: EvalReport,
    extra_json: Optional[Dict] = None,
    csv_name: str = "metrics_summary.csv",
    json_name: str = "metrics_summary.json",
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Tables
    df = dataframe_metrics_summary(report.canonical, report.bandspace)
    df.to_csv(os.path.join(out_dir, csv_name), index=False)

    # JSON
    bundle = {
        "canonical": asdict(report.canonical) if report.canonical is not None else None,
        "bandspace": asdict(report.bandspace) if report.bandspace is not None else None,
        "reliability": {
            "bin_edges": report.reliability.bin_edges.tolist(),
            "sigma_mean": report.reliability.sigma_mean.tolist(),
            "abs_resid_mean": report.reliability.abs_resid_mean.tolist(),
            "abs_resid_std": report.reliability.abs_resid_std.tolist(),
            "expected_abs_gauss": report.reliability.expected_abs_gauss.tolist(),
            "counts": report.reliability.counts.tolist(),
        } if report.reliability is not None else None,
        "per_lambda_rmse": report.per_lambda_rmse.tolist() if report.per_lambda_rmse is not None else None,
        "wavelength_nm": report.wavelength_nm.tolist() if report.wavelength_nm is not None else None,
    }
    if extra_json:
        bundle.update(extra_json)
    with open(os.path.join(out_dir, json_name), "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)


# ---------- High-level evaluators ----------

def build_canonical_report(
    f_hat: np.ndarray,
    f_true: np.ndarray,
    cr_mse: Optional[np.ndarray] = None,
    lambda_weights: Optional[np.ndarray] = None,
    stratify: Optional[np.ndarray] = None,
    bootstrap_reps: int = 1000,
) -> CanonicalReport:
    """
    Assemble canonical-space metrics with bootstrap CIs (median).
    - SAM (rad)
    - CR-MSE (if provided)
    - Curvature energy
    - Non-negativity violation rate
    """
    # SAM
    sam_b = spectral_angle(torch.from_numpy(f_hat), torch.from_numpy(f_true), radians=True).cpu().numpy()
    sam_med = float(np.median(sam_b))
    sam_ci = _bootstrap_ci_np(sam_b, stratify=stratify, reps=bootstrap_reps)

    # CR-MSE
    if cr_mse is None:
        cr_mse = np.mean((f_hat - f_true) ** 2, axis=1)  # fallback if explicit CR not provided
    cr_med = float(np.median(cr_mse))
    cr_ci = _bootstrap_ci_np(cr_mse, stratify=stratify, reps=bootstrap_reps)

    # Curvature & NVR
    curv_b = curvature_energy(torch.from_numpy(f_hat), lambda_weights=torch.from_numpy(lambda_weights) if lambda_weights is not None else None)
    curv_b = curv_b.cpu().numpy()
    curv_med = float(np.median(curv_b))
    curv_ci = _bootstrap_ci_np(curv_b, stratify=stratify, reps=bootstrap_reps)

    nvr_b = nonneg_violation_rate(torch.from_numpy(f_hat)).cpu().numpy()
    nvr_med = float(np.median(nvr_b))
    nvr_ci = _bootstrap_ci_np(nvr_b, stratify=stratify, reps=bootstrap_reps)

    return CanonicalReport(
        sam_rad_median=sam_med,
        sam_rad_ci=sam_ci,
        cr_mse_median=cr_med,
        cr_mse_ci=cr_ci,
        curvature_median=curv_med,
        curvature_ci=curv_ci,
        nvr_median=nvr_med,
        nvr_ci=nvr_ci,
    )


def _bootstrap_ci_np(values: np.ndarray, reps: int = 1000, stratify: Optional[np.ndarray] = None, alpha: float = 0.05) -> Tuple[float, float]:
    v = values.astype(np.float64).ravel()
    N = v.size
    rng = np.random.default_rng()
    def med(x: np.ndarray) -> float:
        return float(np.median(x))
    if stratify is None:
        boot = np.array([med(v[rng.integers(0, N, size=N)]) for _ in range(reps)], dtype=np.float64)
    else:
        labels = np.asarray(stratify).ravel()
        uniq = np.unique(labels)
        boot = np.empty(reps, dtype=np.float64)
        for i in range(reps):
            parts = []
            for u in uniq:
                idx = np.where(labels == u)[0]
                s = rng.choice(idx, size=idx.size, replace=True)
                parts.append(v[s])
            boot[i] = med(np.concatenate(parts, axis=0))
    return float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1.0 - alpha / 2))


def full_evaluation_report(
    *,
    f_hat: np.ndarray,
    f_true: Optional[np.ndarray] = None,
    wavelength_nm: Optional[np.ndarray] = None,
    rmse_mask: Optional[np.ndarray] = None,
    y_hat: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
    nm_per_step: Optional[float] = None,
    residual_sigma: Optional[Tuple[np.ndarray, np.ndarray]] = None,  # (residual, sigma) for bandspace reliability
    cr_mse: Optional[np.ndarray] = None,
    lambda_weights: Optional[np.ndarray] = None,
    stratify: Optional[np.ndarray] = None,
    bootstrap_reps: int = 1000,
) -> EvalReport:
    """
    Build a complete report object with canonical metrics (if f_true provided),
    band-space metrics (if y_hat/y/R provided), per-λ RMSE and reliability curve.
    """
    # Per-λ RMSE (if f_true present)
    rmse_lambda = None
    if f_true is not None:
        rmse_lambda = per_lambda_rmse(torch.from_numpy(f_hat), torch.from_numpy(f_true),
                                      mask=torch.from_numpy(rmse_mask) if rmse_mask is not None else None).cpu().numpy()

    # Canonical report
    canonical = None
    if f_true is not None:
        canonical = build_canonical_report(
            f_hat=f_hat,
            f_true=f_true,
            cr_mse=cr_mse,
            lambda_weights=lambda_weights,
            stratify=stratify,
            bootstrap_reps=bootstrap_reps,
        )

    # Band-space
    bandspace = None
    reliability = None
    if (y_hat is not None) and (y is not None) and (R is not None):
        bs = evaluate_bandspace(y_hat=y_hat, y=y, R=R, nm_per_step=nm_per_step, residual_sigma=residual_sigma)
        bandspace = BandspaceReport(mae_overall=bs.mae_overall, mae_bins=bs.mae_bins, ece=bs.ece)
        if residual_sigma is not None:
            resid, sigma = residual_sigma
            reliability = reliability_by_sigma(resid, sigma, n_bins=12)

    return EvalReport(
        canonical=canonical,
        bandspace=bandspace,
        per_lambda_rmse=rmse_lambda,
        wavelength_nm=wavelength_nm,
        reliability=reliability,
    )


# ---------- Orchestrated write-to-disk helper ----------

def write_eval_package(
    report: EvalReport,
    out_dir: str,
    *,
    plot_rmse: bool = True,
    plot_bins: bool = True,
    plot_reliability_flag: bool = True,
    extra_json: Optional[Dict] = None,
) -> None:
    """
    Persist CSV/JSON tables and optional figures into `out_dir`.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Save summary tables
    from .reports import save_report_artifacts, dataframe_metrics_summary  # self-import for clarity
    save_report_artifacts(out_dir, report, extra_json=extra_json)

    # Plots
    if plot_rmse and report.per_lambda_rmse is not None and report.wavelength_nm is not None:
        plot_per_lambda_rmse(report.wavelength_nm, report.per_lambda_rmse, title="Per-λ RMSE",
                             out_path=os.path.join(out_dir, "per_lambda_rmse.png"))

    if plot_bins and report.bandspace is not None:
        plot_width_binned_mae(report.bandspace.mae_bins, out_path=os.path.join(out_dir, "width_binned_mae.png"))

    if plot_reliability_flag and report.reliability is not None:
        ece = expected_calibration_error_reliability(report.reliability, weight_by_count=True)
        plot_reliability(report.reliability, title=f"Calibration (ECE={ece:.4f})",
                         out_path=os.path.join(out_dir, "reliability.png"))
