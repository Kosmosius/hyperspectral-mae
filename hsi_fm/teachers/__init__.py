"""Teacher wrappers for gas detection baselines."""

from hsi_fm.teachers.lwir_cmf_glrt import estimate_lwir_teacher
from hsi_fm.teachers.swir_ctmf_imap import estimate_swir_teacher

__all__ = ["estimate_swir_teacher", "estimate_lwir_teacher"]
