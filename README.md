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

## Tests

Run the unit tests with:

```bash
pytest -q
```

## License

See [LICENSE](LICENSE).
