from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple
import json
import numpy as np
import numpy.typing as npt
from scipy.interpolate import BSpline

from .types import CanonicalGrid, SplineBasis, SRFProjectionBasis, NDArrayF
from .hashing import sha256_ndarray, stable_meta_hash


def _trapezoid_weights(x: NDArrayF) -> NDArrayF:
    """Trapezoidal integration weights for a sorted 1-D grid."""
    w = np.empty_like(x)
    dx = np.diff(x)
    w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    w[0] = 0.5 * dx[0]
    w[-1] = 0.5 * dx[-1]
    return w


def _make_open_uniform_knot_vector(start: float, stop: float, degree: int, knot_spacing_nm: float) -> np.ndarray:
    num_spans = int(np.floor((stop - start) / knot_spacing_nm + 1e-9))
    if num_spans < 1:
        num_spans = 1
    # open uniform: endpoints repeated degree+1 times
    inner_knots = np.linspace(start, start + knot_spacing_nm * num_spans, num_spans + 1)
    t_start = np.full(degree + 1, start)
    t_end = np.full(degree + 1, inner_knots[-1])
    knots = np.concatenate([t_start, inner_knots[1:-1], t_end])
    return knots


def build_bspline_basis(
    grid: CanonicalGrid,
    knot_spacing_nm: float = 15.0,
    degree: int = 3,
    orthonormalize: bool = True,
    for_projection: bool = False,
) -> SplineBasis | SRFProjectionBasis:
    """
    Build B-spline basis sampled on the canonical grid.
    If orthonormalize=True, columns are orthonormal under trapezoidal inner product.

    If for_projection=True, returns SRFProjectionBasis suitable for projecting SRFs;
    otherwise returns SplineBasis for decoding f(λ).
    """
    lam = grid.lambdas_nm
    K = lam.size
    knots = _make_open_uniform_knot_vector(lam[0], lam[-1], degree, knot_spacing_nm)
    # number of basis funcs = len(knots) - degree - 1
    M = len(knots) - degree - 1
    if M <= 0:
        raise ValueError("Invalid knot vector / degree")

    # Build basis matrix B[k,m] = B_m(λ_k)
    B = np.zeros((K, M), dtype=np.float64)
    eye = np.eye(M, dtype=np.float64)
    for m in range(M):
        coeff = eye[m]
        spl = BSpline(knots, coeff, degree, extrapolate=False)
        B[:, m] = np.nan_to_num(spl(lam), nan=0.0, posinf=0.0, neginf=0.0)

    # Optional orthonormalization under trapezoid inner product
    cond = np.nan
    if orthonormalize:
        W = np.diag(_trapezoid_weights(lam))  # [K,K]
        G = B.T @ W @ B                        # [M,M] Gram
        # Regularize slightly to improve conditioning on dense bases
        eps = 1e-10 * np.trace(G) / M
        G = G + eps * np.eye(M)
        # Cholesky might fail if not PD; use eigendecomp
        eigvals, eigvecs = np.linalg.eigh(G)
        if np.any(eigvals <= 0):
            # fallback to SVD whitening
            U, S, Vt = np.linalg.svd(B, full_matrices=False)
            # Orthonormal columns under L2 ~ W^0.5; approximate under trapezoid
            B = U
            cond = float(S.max() / max(S.min(), 1e-12))
        else:
            G_inv_half = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            B = B @ G_inv_half
            cond = float(eigvals.max() / max(eigvals.min(), 1e-18))

    sha = sha256_ndarray(B)
    if for_projection:
        return SRFProjectionBasis(S=B, degree=degree, knots_nm=knots, grid_name=grid.name, sha256=sha)
    else:
        return SplineBasis(B=B, knots_nm=knots, degree=degree, orthonormal=True, cond_number=cond,
                           grid_name=grid.name, sha256=sha)


def save_basis(basis: SplineBasis | SRFProjectionBasis, path: str | Path) -> None:
    p = Path(path)
    meta = {
        "type": basis.__class__.__name__,
        "degree": basis.degree,
        "grid_name": basis.grid_name,
        "sha256": basis.sha256,
    }
    if isinstance(basis, SplineBasis):
        np.savez_compressed(
            p,
            M=basis.B,
            knots_nm=basis.knots_nm,
            degree=np.array([basis.degree], dtype=np.int32),
            orthonormal=np.array([basis.orthonormal], dtype=np.int8),
            cond=np.array([basis.cond_number], dtype=np.float64),
            meta=json.dumps(meta).encode("utf-8"),
        )
    else:
        np.savez_compressed(
            p,
            S=basis.S,
            knots_nm=basis.knots_nm,
            degree=np.array([basis.degree], dtype=np.int32),
            meta=json.dumps(meta).encode("utf-8"),
        )


def load_basis(path: str | Path) -> SplineBasis | SRFProjectionBasis:
    p = Path(path)
    with np.load(p, allow_pickle=False) as z:
        meta = json.loads(bytes(z["meta"]).decode("utf-8"))
        btype = meta.get("type", "")
        if "M" in z and btype in ("SplineBasis", ""):
            B = z["M"]
            return SplineBasis(
                B=B,
                knots_nm=z["knots_nm"],
                degree=int(z["degree"][0]),
                orthonormal=bool(int(z.get("orthonormal", np.array([1]))[0])),
                cond_number=float(z.get("cond", np.array([np.nan]))[0]),
                grid_name=meta.get("grid_name", "unknown"),
                sha256=meta.get("sha256", sha256_ndarray(B)),
            )
        elif "S" in z:
            S = z["S"]
            return SRFProjectionBasis(
                S=S,
                knots_nm=z["knots_nm"],
                degree=int(z["degree"][0]),
                grid_name=meta.get("grid_name", "unknown"),
                sha256=meta.get("sha256", sha256_ndarray(S)),
            )
        else:
            raise ValueError("Unrecognized basis file")


def project_onto_basis(f_lambda: NDArrayF, basis: SplineBasis) -> NDArrayF:
    """
    Project canonical spectrum onto orthonormal basis: w = B^T W f (but if columns orthonormal under W, w ≈ B^T f*).
    Given orthonormalization used W, B^T W B = I ⇒ least-squares coefficients w = (B^T W B)^{-1} B^T W f = B^T W f.
    For practical use on uniform grids, we approximate using trapezoid weights W.
    """
    if f_lambda.ndim != 1 or f_lambda.shape[0] != basis.B.shape[0]:
        raise ValueError("Dimension mismatch in project_onto_basis")
    W = np.diag(_trapezoid_weights(np.arange(f_lambda.size)))  # approximate W by uniform spacing
    return basis.B.T @ (W @ f_lambda)


def reconstruct_from_basis(w: NDArrayF, basis: SplineBasis) -> NDArrayF:
    """
    Reconstruct spectrum f_hat = B w.
    """
    if w.ndim != 1 or w.shape[0] != basis.B.shape[1]:
        raise ValueError("Coefficient length mismatch in reconstruct_from_basis")
    return basis.B @ w
