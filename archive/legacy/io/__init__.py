"""
I/O package: manifests, datasets, and unified loaders/collate for SpectralAE.

This package is torch-optional:
- Dataset classes implement __len__/__getitem__ and return NumPy arrays + metadata.
- Collate utilities can emit either NumPy batches or PyTorch tensors (if torch is installed).

Assumptions (aligned with the research doc):
- Lab/field spectra are already canonicalized to the CanonicalGrid (e.g., 400–2500 nm @ 3 nm).
- Satellite cubes come with mission SRF matrices precomputed for that CanonicalGrid (R in NPZ).
- Each dataset folder includes a manifest.json that conforms to the schemas defined here.
"""

from .manifests import (
    ManifestError,
    BaseManifest,
    LabManifest,
    FieldManifest,
    SatelliteManifest,
    load_manifest,
    validate_manifest_file,
)
from .datasets.lab import LabLibraryDataset
from .datasets.field import FieldDataset
from .datasets.satellite import SatellitePixelDataset, PRISMA, EnMAP
from .loaders import (
    RaggedBatch,
    collate_samples,
    get_dataloader,        # thin torch wrapper if torch is installed
    TORCH_AVAILABLE,
)

__all__ = [
    # manifests
    "ManifestError", "BaseManifest", "LabManifest", "FieldManifest", "SatelliteManifest",
    "load_manifest", "validate_manifest_file",
    # datasets
    "LabLibraryDataset", "FieldDataset", "SatellitePixelDataset", "PRISMA", "EnMAP",
    # loaders
    "RaggedBatch", "collate_samples", "get_dataloader", "TORCH_AVAILABLE",
]
