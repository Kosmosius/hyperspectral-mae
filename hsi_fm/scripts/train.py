"""Unified training script dispatching the three stages."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from hsi_fm.data.datasets import (
    MAESyntheticOverheadDataset,
    LabSpectraSynthetic,
    OverheadSynthetic,
    collate_lab,
    collate_mae,
    collate_overhead,
)
from hsi_fm.data.patches import PatchConfig
from hsi_fm.model.backbone import HyperspectralMAE, MaskingConfig
from hsi_fm.physics.renderer import LabToSensorRenderer
from hsi_fm.physics.srf import canonical_grid
from hsi_fm.train import StageAExperiment, StageBExperiment, StageCExperiment
from hsi_fm.train.stage_B_align import StageBLossConfig


def _resolve_config(path: Path, overrides: list[str]) -> Any:
    path = path.resolve()
    overrides = list(overrides)
    with initialize_config_dir(config_dir=str(path.parent), job_name="train"):
        cfg = compose(config_name=path.name, overrides=overrides)
        if not any(item.startswith("train=") for item in overrides):
            stage_name = cfg.train.stage
            cfg = compose(
                config_name=path.name,
                overrides=overrides + [f"train={stage_name}"],
            )
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the hyperspectral foundation model")
    parser.add_argument("config", type=str, help="Hydra config file")
    parser.add_argument("overrides", nargs="*", help="Optional Hydra-style overrides")
    args = parser.parse_args()

    cfg = _resolve_config(Path(args.config), args.overrides)

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    stage = cfg.train.stage
    if stage == "stage_a":
        _run_stage_a(cfg)
    elif stage == "stage_b":
        _run_stage_b(cfg)
    elif stage == "stage_c":
        raise NotImplementedError(
            "hook up real loaders or synthetic alignment/gas datasets"
        )
    else:
        raise ValueError(f"Unknown stage {stage}")


def _create_model(cfg: Any) -> HyperspectralMAE:
    masking_cfg = cfg.model.masking
    return HyperspectralMAE(
        embed_dim=cfg.model.embed_dim,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        masking=MaskingConfig(
            spatial_ratio=masking_cfg.spatial_ratio,
            spectral_ratio=masking_cfg.spectral_ratio,
        ),
    )


def _run_stage_a(cfg: Any) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[stage_a] using device: {device}")

    model = _create_model(cfg)

    patch_cfg = PatchConfig(
        height=cfg.data.patch.height,
        width=cfg.data.patch.width,
        stride=cfg.data.patch.stride,
    )
    dataset = MAESyntheticOverheadDataset(
        length=cfg.data.synthetic.length,
        channels=cfg.data.synthetic.channels,
        image_size=cfg.data.synthetic.image_size,
        patch_config=patch_cfg,
        seed=cfg.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_mae,
    )

    experiment = StageAExperiment(model, token_dim=dataset.token_dim, device=device)
    if cfg.train.optimizer.name.lower() != "adamw":
        raise ValueError("Only AdamW optimizer is currently supported for Stage A")
    optimizer = torch.optim.AdamW(
        experiment.parameters(),
        lr=cfg.train.optimizer.lr,
        weight_decay=cfg.train.optimizer.weight_decay,
    )
    experiment.set_optimizer(optimizer)

    max_steps = min(len(dataloader), int(cfg.train.max_steps))
    precision = cfg.train.get("precision", "fp32")
    autocast_dtype = None
    if precision.lower() == "bf16" and device.type == "cuda":
        autocast_dtype = torch.bfloat16

    epoch_loss = experiment.train_epoch(
        dataloader,
        max_steps=max_steps,
        autocast_dtype=autocast_dtype,
        print_shapes=2,
    )
    print(f"[stage_a] epoch_loss={epoch_loss:.4f}")


def _run_stage_b(cfg: Any) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[stage_b] using device: {device}")

    model = _create_model(cfg)

    lab_cfg = cfg.data.synthetic.lab
    overhead_cfg = cfg.data.synthetic.overhead
    hi_grid = canonical_grid(lab_cfg.min_nm, lab_cfg.max_nm, lab_cfg.step_nm)
    sensor_centers = torch.linspace(
        float(hi_grid.min()), float(hi_grid.max()), overhead_cfg.sensor_bands
    )
    sensor_fwhm = torch.full((overhead_cfg.sensor_bands,), overhead_cfg.fwhm_nm)
    physics_params = OmegaConf.to_container(cfg.physics, resolve=True)
    renderer = LabToSensorRenderer(
        hi_wvl_nm=hi_grid,
        sensor_centers_nm=sensor_centers,
        sensor_fwhm_nm=sensor_fwhm,
        mode="swir_reflectance_to_radiance",
        rt_params=dict(physics_params),
    )

    lab_dataset = LabSpectraSynthetic(
        length=lab_cfg.length,
        hi_wvl_nm=hi_grid,
        num_classes=lab_cfg.num_classes,
        seed=cfg.seed,
    )
    overhead_dataset = OverheadSynthetic(
        length=overhead_cfg.length,
        lab_dataset=lab_dataset,
        renderer=renderer,
        seed=cfg.seed + 1234,
        tau_jitter=overhead_cfg.tau_jitter,
        scale_jitter=overhead_cfg.scale_jitter,
        noise_std=overhead_cfg.noise_std,
    )

    lab_loader = DataLoader(
        lab_dataset,
        batch_size=cfg.train.batch_size_lab,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_lab,
    )
    overhead_loader = DataLoader(
        overhead_dataset,
        batch_size=cfg.train.batch_size_overhead,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_overhead,
    )

    loss_cfg = StageBLossConfig(
        recon=cfg.train.loss.recon,
        nce=cfg.train.loss.nce,
        cycle=cfg.train.loss.cycle,
    )

    experiment = StageBExperiment(
        model,
        renderer,
        lab_dim=hi_grid.numel(),
        sensor_dim=sensor_centers.numel(),
        proj_dim=cfg.train.proj_dim,
        device=device,
        loss_cfg=loss_cfg,
        temperature=cfg.train.temperature,
    )

    if cfg.train.optimizer.name.lower() != "adamw":
        raise ValueError("Only AdamW optimizer is currently supported for Stage B")
    optimizer = torch.optim.AdamW(
        experiment.parameters(),
        lr=cfg.train.optimizer.lr,
        weight_decay=cfg.train.optimizer.weight_decay,
    )
    experiment.set_optimizer(optimizer)

    precision = cfg.train.get("precision", "fp32")
    autocast_dtype = None
    if precision.lower() == "bf16" and device.type == "cuda":
        autocast_dtype = torch.bfloat16

    max_steps = cfg.train.get("max_steps")
    epoch_loss = experiment.train_epoch(
        lab_loader,
        overhead_loader,
        max_steps=max_steps,
        autocast_dtype=autocast_dtype,
    )
    print(f"[stage_b] epoch_loss={epoch_loss:.4f}")


if __name__ == "__main__":  # pragma: no cover
    main()
