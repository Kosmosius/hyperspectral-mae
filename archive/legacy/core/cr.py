from __future__ import annotations

from typing import Tuple
import numpy as np
import numpy.typing as npt
from scipy.signal import savgol_filter


NDArrayF = npt.NDArray[np.floating]


def _upper_convex_hull(x: NDArrayF, y: NDArrayF) -> Tuple[NDArrayF, NDArrayF]:
    """
    Compute the upper convex hull of (x, y) using a monotone chain (Andrew's algorithm).
    x must be strictly increasing. Returns hull points (xh, yh), both increasing in x.
    """
    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be strictly increasing")

    pts = np.stack([x, y], axis=1)

    def cross(o, a, b):
        # cross product of OA × OB; positive if OAB makes a left turn
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    upper = []
    for p in pts:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) >= 0:  # >=0 removes concave or collinear downward
            upper.pop()
        upper.append(p)
    upper = np.array(upper)
    # Filter to ensure monotone x
    _, idx = np.unique(upper[:, 0], return_index=True)
    upper = upper[np.sort(idx)]
    return upper[:, 0], upper[:, 1]


def convex_hull_continuum(lambdas_nm: NDArrayF, f_lambda: NDArrayF) -> NDArrayF:
    """
    Classical convex-hull continuum (Clark & Roush 1984). Returns continuum c(λ) ≥ f(λ).
    We build the upper convex hull and linearly interpolate between hull vertices.
    """
    if lambdas_nm.ndim != 1 or f_lambda.ndim != 1 or lambdas_nm.size != f_lambda.size:
        raise ValueError("Inputs must be 1-D arrays of equal length")
    x = lambdas_nm.astype(np.float64)
    y = np.maximum(f_lambda.astype(np.float64), 0.0)

    xh, yh = _upper_convex_hull(x, y)

    # Interpolate hull to all x locations
    c = np.interp(x, xh, yh, left=y[0], right=y[-1])
    # Ensure continuum >= signal
    c = np.maximum(c, y)
    return c


def continuum_removed(lambdas_nm: NDArrayF, f_lambda: NDArrayF, eps: float = 1e-6) -> NDArrayF:
    """
    Continuum-removed spectrum: CR = f / c, clipped to [0, 1.2] (small slack for numeric noise).
    """
    c = convex_hull_continuum(lambdas_nm, f_lambda)
    cr = f_lambda / np.clip(c, eps, None)
    return np.clip(cr, 0.0, 1.2)


def smoothed_continuum_removed(
    lambdas_nm: NDArrayF,
    f_lambda: NDArrayF,
    window: int = 31,
    polyorder: int = 3,
    eps: float = 1e-6,
) -> NDArrayF:
    """
    Training-friendly CR with smoothed continuum (Savitzky–Golay on the hull).
    """
    c_hull = convex_hull_continuum(lambdas_nm, f_lambda)
    # Adjust window to be odd and within bounds
    window = max(5, int(window) | 1)
    if window >= f_lambda.size:
        window = f_lambda.size - 1 if (f_lambda.size % 2 == 0) else f_lambda.size
    c_smooth = savgol_filter(c_hull, window_length=window, polyorder=min(polyorder, window - 1), mode="interp")
    cr = f_lambda / np.clip(c_smooth, eps, None)
    return np.clip(cr, 0.0, 1.2)
