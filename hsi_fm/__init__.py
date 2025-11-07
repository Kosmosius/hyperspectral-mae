"""Top-level package for the hyperspectral foundation model."""
from importlib import metadata

from hsi_fm.model import (
    FactorizedBlock,
    GasPlumeHead,
    HyperspectralMAE,
    LabPrototypeHead,
    MaskingConfig,
    SensorEmbedding,
    SolidsUnmixingHead,
    TubePatchEmbed3D,
    dice_loss,
    mae_loss,
    sam_loss,
)
from hsi_fm.physics import LabToSensorRenderer, SpectralResponseFunction, convolve_srf, gaussian_srf
from hsi_fm.train import (
    EMA,
    EMAConfig,
    StageAConfig,
    StageAExperiment,
    StageBConfig,
    StageBExperiment,
    StageCConfig,
    StageCExperiment,
    TrainingLoop,
    LoopConfig,
    dataloader_worker_seed,
    set_seed,
)

__all__ = [
    "__version__",
    "HyperspectralMAE",
    "MaskingConfig",
    "FactorizedBlock",
    "TubePatchEmbed3D",
    "SensorEmbedding",
    "GasPlumeHead",
    "LabPrototypeHead",
    "SolidsUnmixingHead",
    "mae_loss",
    "sam_loss",
    "dice_loss",
    "SpectralResponseFunction",
    "convolve_srf",
    "gaussian_srf",
    "LabToSensorRenderer",
    "EMA",
    "EMAConfig",
    "StageAExperiment",
    "StageAConfig",
    "StageBExperiment",
    "StageBConfig",
    "StageCExperiment",
    "StageCConfig",
    "TrainingLoop",
    "LoopConfig",
    "set_seed",
    "dataloader_worker_seed",
]


def _get_version() -> str:
    try:
        return metadata.version("hsi-fm")
    except metadata.PackageNotFoundError:  # pragma: no cover - fallback for dev installs
        return "0.0.0"


__version__ = _get_version()
