"""Unified training script dispatching the three stages."""
from __future__ import annotations

import argparse

from omegaconf import OmegaConf

from hsi_fm.train import StageAExperiment, StageBExperiment, StageCExperiment


STAGES = {
    "stage_a": StageAExperiment,
    "stage_b": StageBExperiment,
    "stage_c": StageCExperiment,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the hyperspectral foundation model")
    parser.add_argument("config", type=str, help="Hydra config file")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    stage = cfg.train.stage
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage}")
    print(f"Configured stage: {stage}")


if __name__ == "__main__":  # pragma: no cover
    main()
