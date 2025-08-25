"""
Universal Spectral Representation (USR): tokenization utilities and RFF banks.

Tokens per band:
  t_i = [ y_i | RFF(c_i) | RFF(Δ_i) | s_i ]
where s_i are optional SRF shape projections.

Design goals:
- Pure NumPy (no torch) so it can be used in data pipelines and sanity checks.
- Reproducible RFF banks (seeded & persisted).
- Flexible: accept explicit (c, Δ), or derive from SRF matrix R and CanonicalGrid.
"""

from .rff import RFFBank, RFFSpec, save_rff_bank, load_rff_bank
from .tokenizer import (
    TokenizerConfig,
    Tokenizer,
    TokenizeResult,
    compute_centers_widths_from_R,
)

__all__ = [
    "RFFBank", "RFFSpec", "save_rff_bank", "load_rff_bank",
    "TokenizerConfig", "Tokenizer", "TokenizeResult",
    "compute_centers_widths_from_R",
]
