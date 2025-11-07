"""Evaluation driver."""
from __future__ import annotations

import argparse

from omegaconf import OmegaConf

from hsi_fm.eval import evaluate_emit_min, evaluate_methane_mdl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the hyperspectral foundation model")
    parser.add_argument("config", type=str)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    print(f"Evaluating solids with metrics: {cfg.eval.solids.metrics}")
    print(f"Evaluating gases with metrics: {cfg.eval.gases.metrics}")


if __name__ == "__main__":  # pragma: no cover
    main()
