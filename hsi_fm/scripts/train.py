"""Unified training script dispatching the three stages."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from hydra import compose, initialize
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from hsi_fm import HyperspectralMAE, MaskingConfig
from hsi_fm.train import (
    EMA,
    LoopConfig,
    StageAExperiment,
    TrainingLoop,
    dataloader_worker_seed,
    set_seed,
)


@dataclass
class SyntheticDatasetConfig:
    """Configuration for the synthetic Stage A dataset used in tests."""

    tokens: int = 64
    features: int = 128


class SyntheticStageADataset(Dataset):
    """Generates random token/target pairs for quick smoke tests."""

    def __init__(self, length: int, cfg: SyntheticDatasetConfig):
        self.length = int(length)
        self.cfg = cfg

    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # pragma: no cover - simple random access
        generator = torch.Generator().manual_seed(idx)
        tokens = torch.randn(self.cfg.tokens, self.cfg.features, generator=generator)
        target = torch.randn(self.cfg.tokens, self.cfg.features, generator=generator)
        return {"tokens": tokens, "target": target}


def _load_config(args: argparse.Namespace) -> DictConfig:
    script_dir = Path(__file__).resolve().parent
    if args.config_path and args.config_name:
        search_path = Path(args.config_path).resolve()
        config_name = args.config_name
    elif args.config:
        config_file = Path(args.config).resolve()
        search_path = config_file.parent
        config_name = config_file.name
    else:  # pragma: no cover - CLI guard
        raise ValueError("Either --config-path/--config-name or a config file must be provided")

    relative_path = os.path.relpath(search_path, script_dir)
    with initialize(version_base=None, config_path=relative_path):
        cfg = compose(config_name=config_name)
    return cfg


def _build_model(cfg: DictConfig) -> HyperspectralMAE:
    masking = MaskingConfig(
        spatial_ratio=float(cfg.model.masking.spatial_ratio),
        spectral_ratio=float(cfg.model.masking.spectral_ratio),
    )
    model = HyperspectralMAE(
        embed_dim=int(cfg.model.embed_dim),
        depth=int(cfg.model.depth),
        num_heads=int(cfg.model.num_heads),
        mlp_ratio=float(cfg.model.get("mlp_ratio", 4.0)),
        masking=masking,
    )
    return model


def _run_stage_a(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(cfg)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.optimizer.lr),
        weight_decay=float(cfg.train.optimizer.get("weight_decay", 0.0)),
    )
    ema = EMA(model)
    experiment = StageAExperiment(model, optimizer, ema=ema)

    dataset_cfg = SyntheticDatasetConfig(
        tokens=int(cfg.data.get("tokens", 64)),
        features=model.embed_dim,
    )
    steps = min(int(cfg.train.max_steps), 8)
    dataset = SyntheticStageADataset(length=steps * int(cfg.train.batch_size), cfg=dataset_cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=0,
        worker_init_fn=dataloader_worker_seed,
    )
    loop = TrainingLoop(experiment, dataloader, LoopConfig(max_steps=steps))
    avg_loss = loop.run()
    print(f"Stage A completed {steps} steps with avg loss {avg_loss:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the hyperspectral foundation model")
    parser.add_argument("config", nargs="?", help="Path to a Hydra config file")
    parser.add_argument("--config-path", dest="config_path", help="Directory containing configs")
    parser.add_argument("--config-name", dest="config_name", help="Name of the config file")
    args = parser.parse_args()

    cfg = _load_config(args)
    if "seed" in cfg:
        set_seed(int(cfg.seed))

    stage = cfg.train.stage
    print(f"Configured stage: {stage}")
    if stage == "stage_a":
        _run_stage_a(cfg)
    else:
        raise NotImplementedError(f"Training stage '{stage}' is not yet implemented in this demo setup")


if __name__ == "__main__":  # pragma: no cover
    main()
