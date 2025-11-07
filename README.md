# Hyperspectral Foundation Model Scaffold

This repository provides a scaffold for a production-ready hyperspectral foundation model supporting SWIR and LWIR modalities. The implementation follows a three-stage training procedure with dedicated physics adapters for bridging lab spectra and overhead imagery.

## Features

- Hydra-based configuration system with modular data/model/train presets.
- Physics layer with spectral response function convolution and a differentiable lab-to-sensor renderer.
- Masked autoencoder backbone with spectral-spatial attention blocks.
- Task heads for solid unmixing and gas plume detection.
- Teacher wrappers for SWIR (CTMF/IMAP) and LWIR (CMF/GLRT) supervision.
- Training scripts for each stage and consolidated evaluation utilities.
- Unit tests covering SRF convolution and renderer consistency.

## Repository layout

```
hsi_fm/
  configs/    # Hydra configs for data, models, training, and physics
  data/       # Dataset and patch sampling utilities
  eval/       # Lightweight evaluation entry points
  model/      # Backbone, heads, and loss functions
  physics/    # Spectral response and rendering helpers
  teachers/   # Placeholder teacher signals
  train/      # Training utilities and stage drivers
  scripts/    # CLI entry points (train/eval/prepare-data)
  tests/      # Pytest suite targeting the core modules
tools/
  repo_cleanup.py  # Maintains the hsi_fm-centric layout
archive/
  legacy/     # Original modules kept for reference
```

## Installation

```bash
pip install -e .
```

## Quickstart

1. Prepare toy data assets (optional):

   ```bash
   python -m hsi_fm.scripts.prepare_data ./data_stub
   ```

2. Launch Stage A pretraining on synthetic data:

   ```bash
   hsi-train --config-path hsi_fm/configs --config-name config.yaml
   ```

   You can also invoke the module directly:

   ```bash
   python -m hsi_fm.scripts.train hsi_fm/configs/config.yaml
   ```

3. Evaluate the placeholder metrics:

   ```bash
   hsi-eval hsi_fm/configs/config.yaml
   ```

## Real data quickstart

The repository ships only the scaffolding; you must supply EMIT/EnMAP products locally.

1. Configure EMIT L2A ingestion by pointing Hydra to your files:

   ```bash
   python -m hsi_fm.scripts.train \
     hsi_fm/configs/config.yaml \
     train.stage=stage_b \
     data=emit_l2a \
     data.paths='[/path/to/emit/*.nc]' \
     train.batch_size=2
   ```

   The command initialises the Stage B pipeline, parses SRFs, builds geometry/sensor tokens, and exercises the physics alignment loop for a couple of steps. Substitute `emit_l1b` or `enmap_l2a` to ingest other products. QA filtering thresholds and tiling dimensions live in the corresponding `hsi_fm/configs/data/*.yaml` files.

2. Evaluate EMIT L2B mineral predictions produced by an external model:

   ```bash
   hsi-evaluate-emit-min --pred ./predictions --truth '/data/emit_l2b/*.npz' --out metrics.csv
   ```

   Both predictions and truth are expected to be `.npz` archives containing `mineral_ids`, `band_depths`, `reflectance`, and `wavelengths` arrays.

### Distributed and mixed precision training

- Toggle DistributedDataParallel via `train.ddp=true` and launch with `torchrun`/`torch.distributed.run`. Checkpoints are only emitted on rank 0.
- Automatic mixed precision is controlled with `train.precision` (`fp32`, `bf16`, `fp16`).
- FlashAttention v2 can be enabled with `train.use_flash=true` on PyTorch ≥ 2.2; the script falls back gracefully if kernels are unavailable.
- Gradient checkpointing is exposed through `train.use_checkpointing`.

> **Note:** The repository performs CPU-only execution for CI. GPU execution is supported when CUDA devices are present.

## Tests

Run the unit tests with:

```bash
pytest -q
```

## License

See [LICENSE](LICENSE).
