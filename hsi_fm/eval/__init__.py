"""Evaluation utilities for the hyperspectral foundation model."""

from hsi_fm.eval.emit_min_eval import evaluate_emit_min
from hsi_fm.eval.methane_mdl_eval import evaluate_methane_mdl

__all__ = ["evaluate_emit_min", "evaluate_methane_mdl"]
