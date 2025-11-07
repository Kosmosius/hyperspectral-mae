"""Unified training script dispatching the three stages."""
from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.distributed as dist
from hydra import compose, initialize
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from hsi_fm import HyperspectralMAE, MaskingConfig
from hsi_fm.data.emit import load_emit_srf_table
from hsi_fm.data.enmap import load_enmap_srf_table
from hsi_fm.data.overhead import OverheadHSI, collate_overhead
from hsi_fm.data.patches import PatchConfig
from hsi_fm.train import EMA, StageAExperiment, dataloader_worker_seed, set_seed
from hsi_fm.utils.dist import get_rank, init_distributed, is_distributed, is_main_process


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


def _load_config(args: argparse.Namespace, overrides: List[str]) -> DictConfig:
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
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _build_model(cfg: DictConfig, device: torch.device) -> HyperspectralMAE:
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
    if bool(cfg.train.get("use_checkpointing", False)) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.to(device)
    return model


def _setup_precision(precision: str, device: torch.device) -> torch.dtype | None:
    precision = precision.lower()
    if precision == "fp16":
        return torch.float16 if device.type == "cuda" else None
    if precision == "bf16":
        return torch.bfloat16 if device.type in {"cuda", "cpu"} else None
    return None


def _configure_flash_attention(enabled: bool) -> None:
    if not enabled:
        return
    try:
        torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False)
        if is_main_process():
            print("Enabled FlashAttention SDPA backend")
    except AttributeError:
        if is_main_process():
            print("FlashAttention kernels unavailable; falling back to default")


def _select_device(local_rank: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", index=local_rank)
    return torch.device("cpu")


def _enumerate_files(patterns: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        files.extend(Path(p) for p in glob.glob(pattern))
    return sorted(files)


def _load_srf_table(data_cfg: DictConfig):
    table_path = data_cfg.get("srf_table")
    if not table_path:
        return None
    path = Path(table_path)
    if data_cfg.source.startswith("emit"):
        return load_emit_srf_table(path)
    return load_enmap_srf_table(path)


def _build_overhead_dataset(cfg: DictConfig) -> OverheadHSI:
    files = _enumerate_files(cfg.paths)
    if not files:
        raise ValueError("No files matched the provided data paths")
    patch_cfg = PatchConfig(
        height=int(cfg.patch.height),
        width=int(cfg.patch.width),
        stride=int(cfg.patch.stride),
    )
    reader_kwargs: Dict = {}
    table = _load_srf_table(cfg)
    if table is not None:
        reader_kwargs["srf_table"] = table
    min_fraction = float(cfg.qa.get("min_fraction", 0.5)) if "qa" in cfg else 0.5
    return OverheadHSI(
        source=str(cfg.source),
        paths=files,
        patch_config=patch_cfg,
        min_valid_fraction=min_fraction,
        **reader_kwargs,
    )


def _sensor_tokens(sensors: List[str]) -> torch.Tensor:
    lookup = {name: idx for idx, name in enumerate(sorted(set(sensors)))}
    return torch.tensor([lookup[name] for name in sensors], dtype=torch.long)


def _geometry_tokens(geometry: Dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
    if not geometry:
        return torch.zeros(batch_size, 0, dtype=torch.float32)
    keys = sorted(geometry.keys())
    stacked = torch.stack([geometry[key].float() for key in keys], dim=-1)
    return stacked


def _prepare_stage_b_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    tokens = batch["tokens"].mean(dim=1)
    geometry = _geometry_tokens(batch.get("geometry", {}), tokens.shape[0])
    result = {
        "lab_reflectance": tokens,
        "wavelengths": batch.get("wavelengths"),
        "wavelength_mask": batch.get("wavelength_mask"),
        "srf": batch.get("srf"),
        "srf_mask": batch.get("srf_mask"),
        "sensor_tokens": _sensor_tokens(batch.get("sensor", [])),
        "geometry_tokens": geometry,
    }
    return result


def _run_stage_a(cfg: DictConfig, device: torch.device, local_rank: int) -> None:
    model = _build_model(cfg, device)
    if bool(cfg.train.get("ddp", False)):
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.optimizer.lr),
        weight_decay=float(cfg.train.optimizer.get("weight_decay", 0.0)),
    )
    ema = EMA(model)
    experiment = StageAExperiment(model, optimizer, ema=ema)

    dataset_cfg = SyntheticDatasetConfig(
        tokens=int(cfg.data.get("tokens", 64)),
        features=model.module.embed_dim if isinstance(model, DistributedDataParallel) else model.embed_dim,
    )
    steps = min(int(cfg.train.max_steps), 8)
    dataset = SyntheticStageADataset(length=steps * int(cfg.train.batch_size), cfg=dataset_cfg)
    sampler = None
    if is_distributed():
        sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=get_rank())
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        worker_init_fn=dataloader_worker_seed,
    )
    try:
        scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    except AttributeError:  # pragma: no cover - compatibility for older torch builds
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    model.train()
    total_loss = 0.0
    clip_grad = float(cfg.train.get("clip_grad_norm", 1.0))
    accum = int(cfg.train.get("accumulate_grad_batches", 1))
    optimizer.zero_grad(set_to_none=True)
    autocast_dtype = _setup_precision(cfg.train.get("precision", "fp32"), device)
    for step, batch in enumerate(dataloader):
        if step >= steps:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            loss = experiment.training_step(batch) / accum
        scaler.scale(loss).backward()
        if (step + 1) % accum == 0:
            scaler.unscale_(optimizer)
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if experiment.ema is not None and is_main_process():
                experiment.ema.update(model)
        total_loss += float(loss.detach())
    avg_loss = total_loss / max(steps, 1)
    if is_main_process():
        print(f"Stage A completed {steps} steps with avg loss {avg_loss:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the hyperspectral foundation model")
    parser.add_argument("config", nargs="?", help="Path to a Hydra config file")
    parser.add_argument("--config-path", dest="config_path", help="Directory containing configs")
    parser.add_argument("--config-name", dest="config_name", help="Name of the config file")
    args, overrides = parser.parse_known_args()
    if args.config and "=" in args.config:
        overrides.insert(0, args.config)
        args.config = None

    cfg = _load_config(args, overrides)
    if "seed" in cfg:
        set_seed(int(cfg.seed))

    stage = cfg.train.stage
    ddp = bool(cfg.train.get("ddp", False))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if ddp:
        init_distributed(cfg.train.get("backend"))
    device = _select_device(local_rank)
    _configure_flash_attention(bool(cfg.train.get("use_flash", False)))

    if stage == "stage_a":
        _run_stage_a(cfg, device, local_rank)
    elif stage == "stage_b":
        dataset = _build_overhead_dataset(cfg.data)
        sampler = None
        if ddp:
            sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=get_rank())
        dataloader = DataLoader(
            dataset,
            batch_size=int(cfg.train.batch_size),
            shuffle=sampler is None,
            sampler=sampler,
            collate_fn=collate_overhead,
            num_workers=0,
        )
        for step, batch in enumerate(dataloader):
            prepared = _prepare_stage_b_batch(batch)
            if is_main_process() and step == 0:
                print(
                    "Loaded Stage B batch",
                    {"lab_reflectance": prepared["lab_reflectance"].shape, "sensor_tokens": prepared["sensor_tokens"].shape},
                )
            if step >= 2:
                break
        if is_main_process():
            print("Stage B initialisation completed")
    else:
        raise NotImplementedError(f"Training stage '{stage}' is not yet implemented in this demo setup")


if __name__ == "__main__":  # pragma: no cover
    main()
