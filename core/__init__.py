"""
Physics & numerics core for SpectralAE (torch-free).

This package defines:
- Data models & types (CanonicalGrid, SRF, SRFMatrix, Sample, Basis).
- Canonical wavelength grids, masks, and validation.
- SRF curves, parametric surrogates, projection basis utilities.
- High-precision renderer utilities (SRF→R matrix; Gauss–Legendre quadrature).
- B-spline basis construction with orthonormalization & versioning.
- Continuum removal (classical convex-hull and smoothed variant).
- SHA-256 hashing utilities for provenance tracking.

All modules here must remain torch-agnostic.
"""

from .types import (
    CanonicalGrid,
    SRF,
    SRFMatrix,
    Sample,
    SplineBasis,
    SRFProjectionBasis,
)
from .grid import create_grid, load_grid_csv, save_grid_csv, build_water_vapor_mask
from .srf import (
    gaussian_srf,
    triangular_srf,
    box_srf,
    skewed_gaussian_srf,
    normalize_srf_curve,
    build_srf_matrix_from_curves,
    project_srf_matrix_onto_basis,
)
from .renderer import (
    build_R_discrete,
    build_R_gauss_legendre,
    render_bands,
    verify_row_stochastic,
    effective_width_bins,
    effective_width_nm,
    save_R_npz,
    load_R_npz,
)
from .basis import (
    build_bspline_basis,
    save_basis,
    load_basis,
    project_onto_basis,
    reconstruct_from_basis,
)
from .cr import (
    convex_hull_continuum,
    continuum_removed,
    smoothed_continuum_removed,
)
from .hashing import sha256_bytes, sha256_ndarray, sha256_path, stable_meta_hash

__all__ = [
    # types
    "CanonicalGrid", "SRF", "SRFMatrix", "Sample", "SplineBasis", "SRFProjectionBasis",
    # grid
    "create_grid", "load_grid_csv", "save_grid_csv", "build_water_vapor_mask",
    # srf
    "gaussian_srf", "triangular_srf", "box_srf", "skewed_gaussian_srf",
    "normalize_srf_curve", "build_srf_matrix_from_curves", "project_srf_matrix_onto_basis",
    # renderer
    "build_R_discrete", "build_R_gauss_legendre", "render_bands", "verify_row_stochastic",
    "effective_width_bins", "effective_width_nm", "save_R_npz", "load_R_npz",
    # basis
    "build_bspline_basis", "save_basis", "load_basis", "project_onto_basis", "reconstruct_from_basis",
    # continuum removal
    "convex_hull_continuum", "continuum_removed", "smoothed_continuum_removed",
    # hashing
    "sha256_bytes", "sha256_ndarray", "sha256_path", "stable_meta_hash",
]
