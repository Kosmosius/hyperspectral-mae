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

## Installation

```bash
pip install -e .
```

## Quickstart

1. Prepare toy data assets:

```bash
python -m hsi_fm.scripts.prepare_data ./data_stub
```

2. Launch pretraining using the default configuration:

```bash
python -m hsi_fm.scripts.train hsi_fm/configs/config.yaml
```

3. Evaluate on placeholder benchmarks:

```bash
python -m hsi_fm.scripts.evaluate hsi_fm/configs/config.yaml
```

## Tests

Run the unit tests with:

```bash
pytest
```

## License

See [LICENSE](LICENSE).
