from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Tuple
import json
import numpy as np
from numpy.polynomial.legendre import leggauss

from .types import CanonicalGrid, SRF, SRFMatrix, NDArrayF
from .hashing import sha256_ndarray, stable_meta_hash
from .srf import normalize_srf_curve


def _row_normalize_nonneg(M: np.ndarray) -> np.ndarray:
    M = np.maximum(M, 0.0)
    rs = np.sum(M, axis=1, keepdims=True)
    rs = np.where(rs <= 0, 1.0, rs)
    return M / rs


def build_R_discrete(srfs: Iterable[SRF], grid: CanonicalGrid) -> SRFMatrix:
    """
    Fast path: interpolate each SRF to the canonical grid and renormalize per row.
    """
    from .srf import build_srf_matrix_from_curves
    return build_srf_matrix_from_curves(srfs, grid)


def _integrate_band_weights_gauss_legendre(
    srf: SRF,
    grid: CanonicalGrid,
    nodes_per_band: int = 16,
) -> np.ndarray:
    """
    Compute discrete band weights r[K] by integrating SRF over intervals between grid points
    using Gauss–Legendre quadrature mapped to each interval, then renormalize.
    """
    # Normalize SRF curve to unit area on its native support
    s_resp = normalize_srf_curve(srf.wavelength_nm, srf.response)

    # Build piecewise-linear interpolant of SRF for quadrature evaluation
    # Use numpy interp (vectorized); for nodes we sample within each grid interval
    lam = grid.lambdas_nm
    K = lam.size
    r = np.zeros(K, dtype=np.float64)

    # Gauss–Legendre nodes/weights on [-1,1]
    xg, wg = leggauss(nodes_per_band)

    # Integrate the contribution each grid sample receives from its adjacent cell mass
    # Here we apportion integrated SRF mass over bins as if using midpoint control volumes.
    # We compute integral of SRF over cell [λ_k-Δ/2, λ_k+Δ/2], clamped to domain bounds.
    # Edge bins have half cells.
    # Precompute interpolation helper
    def srf_val(x: np.ndarray) -> np.ndarray:
        return np.interp(x, srf.wavelength_nm, s_resp, left=0.0, right=0.0)

    # Compute cell edges
    edges = np.empty(K + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (lam[:-1] + lam[1:])
    edges[0] = lam[0] - 0.5 * (lam[1] - lam[0])
    edges[-1] = lam[-1] + 0.5 * (lam[-1] - lam[-2])

    for k in range(K):
        a = edges[k]
        b = edges[k + 1]
        if b <= srf.wavelength_nm[0] or a >= srf.wavelength_nm[-1]:
            r[k] = 0.0
            continue
        # Map GL nodes from [-1,1] to [a,b]
        xm = 0.5 * (b - a) * xg + 0.5 * (a + b)
        r[k] = 0.5 * (b - a) * np.sum(wg * srf_val(xm))

    # Renormalize to unit sum
    ssum = float(np.sum(r))
    if ssum <= 0:
        # Fallback: delta at nearest center
        idx = int(np.argmin(np.abs(lam - (srf.meta.get("center_nm") or float(np.mean(srf.wavelength_nm))))))
        r[:] = 0.0
        r[idx] = 1.0
    else:
        r /= ssum
    return r


def build_R_gauss_legendre(srfs: Iterable[SRF], grid: CanonicalGrid, nodes_per_band: int = 16) -> SRFMatrix:
    """
    High-accuracy path: compute discrete weights by Gauss–Legendre quadrature per grid bin.
    More expensive than build_R_discrete but better with sharp SRF tails or coarse grids.
    """
    rows = []
    centers = []
    fwhms = []
    for s in srfs:
        r = _integrate_band_weights_gauss_legendre(s, grid, nodes_per_band)
        rows.append(r)
        centers.append(s.meta.get("center_nm", float(np.sum(grid.lambdas_nm * r))))
        # Approximate FWHM via effective width in nm (proxy)
        fwhms.append(float(1.0 / np.sum(r**2) * grid.step_nm))
    R = np.vstack(rows)
    sha = sha256_ndarray(R)
    ew_bins = 1.0 / np.maximum(np.sum(R**2, axis=1), 1e-12)
    return SRFMatrix(
        R=R,
        centers_nm=np.asarray(centers, dtype=np.float64),
        fwhm_nm=np.asarray(fwhms, dtype=np.float64),
        effective_width_bins_=ew_bins,
        grid_name=grid.name,
        sha256=sha,
        meta={"nodes_per_band": np.array([nodes_per_band], dtype=np.int32)},
    )


def render_bands(f_lambda: NDArrayF, R: SRFMatrix) -> NDArrayF:
    """
    Render band-averaged predictions y = R f. Shapes: f[K], R[N,K] ⇒ y[N].
    """
    if f_lambda.ndim != 1 or f_lambda.shape[0] != R.R.shape[1]:
        raise ValueError("f_lambda length must equal SRFMatrix K")
    return R.R @ f_lambda


def verify_row_stochastic(R: SRFMatrix, atol: float = 1e-8) -> bool:
    rowsum = np.sum(R.R, axis=1)
    nonneg = np.all(R.R >= -1e-15)
    return bool(np.allclose(rowsum, 1.0, atol=atol) and nonneg)


def effective_width_bins(R: SRFMatrix) -> NDArrayF:
    """Return effective width in bins: EW = 1 / ∑ r_i^2."""
    return 1.0 / np.maximum(np.sum(R.R**2, axis=1), 1e-12)


def effective_width_nm(R: SRFMatrix, grid: CanonicalGrid) -> NDArrayF:
    """Return effective width in nm: EW_nm = EW_bins * Δλ (approx for uniform grids)."""
    return effective_width_bins(R) * grid.step_nm


def save_R_npz(R: SRFMatrix, path: str | Path) -> None:
    p = Path(path)
    meta = {
        "grid_name": R.grid_name or "",
        "sha256": R.sha256 or sha256_ndarray(R.R),
    }
    np.savez_compressed(
        p,
        R=R.R,
        centers_nm=R.centers_nm if R.centers_nm is not None else np.array([]),
        fwhm_nm=R.fwhm_nm if R.fwhm_nm is not None else np.array([]),
        ew_bins=R.effective_width_bins_ if R.effective_width_bins_ is not None else np.array([]),
        meta=json.dumps(meta).encode("utf-8"),
    )


def load_R_npz(path: str | Path) -> SRFMatrix:
    p = Path(path)
    with np.load(p, allow_pickle=False) as z:
        R = z["R"]
        centers_nm = z["centers_nm"]
        fwhm_nm = z["fwhm_nm"]
        ew_bins = z["ew_bins"]
        meta = json.loads(bytes(z["meta"]).decode("utf-8"))
    return SRFMatrix(
        R=R,
        centers_nm=centers_nm if centers_nm.size > 0 else None,
        fwhm_nm=fwhm_nm if fwhm_nm.size > 0 else None,
        effective_width_bins_=ew_bins if ew_bins.size > 0 else None,
        grid_name=meta.get("grid_name"),
        sha256=meta.get("sha256"),
        meta={},
    )
