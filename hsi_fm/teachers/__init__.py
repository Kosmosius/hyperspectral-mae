"""Teacher wrappers for gas detection baselines."""

from .swir_ctmf_imap import estimate_swir_teacher
from .lwir_cmf_glrt import estimate_lwir_teacher

__all__ = ["estimate_swir_teacher", "estimate_lwir_teacher"]
