"""Dataset abstractions for hyperspectral cubes and lab spectra."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import torch
from torch import Generator, Tensor

if TYPE_CHECKING:  # pragma: no cover
    from hsi_fm.physics.renderer import LabToSensorRenderer
from torch.utils.data import Dataset

from .patches import PatchConfig, PatchSampler


@dataclass
class SceneMetadata:
    """Metadata describing a hyperspectral scene.

    Attributes
    ----------
    sensor: str
        Sensor name (e.g. ``"EMIT"``).
    wavelengths: Tensor
        Wavelength centers in microns.
    sza: float
        Solar zenith angle in degrees.
    vza: float
        View zenith angle in degrees.
    raa: float
        Relative azimuth angle in degrees.
    extra: Mapping[str, float]
        Additional parameters (e.g. aerosol optical thickness).
    """

    sensor: str
    wavelengths: Tensor
    sza: float
    vza: float
    raa: float
    extra: Mapping[str, float]


class HyperspectralCube(Dataset[Mapping[str, torch.Tensor]]):
    """Generic dataset wrapper around hyperspectral cubes.

    The class abstracts away the sensor-specific loading logic. Implementations
    are expected to subclass and override :meth:`_load_cube` to return a tensor
    shaped as ``(C, H, W)`` in radiance or reflectance units.
    """

    def __init__(self, paths: Iterable[Path], metadata: Iterable[SceneMetadata]):
        self._paths = list(paths)
        self._metadata = list(metadata)
        if len(self._paths) != len(self._metadata):
            raise ValueError("paths and metadata must have the same length")

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._paths)

    def __getitem__(self, idx: int) -> Mapping[str, torch.Tensor]:
        cube = self._load_cube(self._paths[idx])
        info = self._metadata[idx]
        return {
            "cube": cube,
            "metadata": {
                "sensor": info.sensor,
                "wavelengths": info.wavelengths.to(dtype=cube.dtype, device=cube.device),
                "angles": torch.tensor([info.sza, info.vza, info.raa], dtype=torch.float32),
                "extra": torch.tensor(list(info.extra.values()), dtype=torch.float32)
                if info.extra
                else torch.zeros(0, dtype=torch.float32),
            },
        }

    def _load_cube(self, path: Path) -> torch.Tensor:
        raise NotImplementedError


class LabSpectraDataset(Dataset[Mapping[str, torch.Tensor]]):
    """Dataset for lab-based reflectance spectra."""

    def __init__(self, spectra: Tensor, labels: Optional[List[str]] = None):
        if spectra.ndim != 2:
            raise ValueError("spectra must be 2D")
        self._spectra = spectra.to(dtype=torch.float32)
        self._labels = labels or [""] * len(spectra)
        if len(self._labels) != len(spectra):
            raise ValueError("labels length mismatch")

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._spectra)

    def __getitem__(self, idx: int) -> Mapping[str, torch.Tensor]:
        return {
            "spectrum": self._spectra[idx],
            "label": self._labels[idx],
        }


class CachedIterableDataset(Dataset[Mapping[str, torch.Tensor]]):
    """Materialized view over an iterator dataset.

    This helper converts an arbitrary iterator of samples into a dataset with
    deterministic random access, which is useful for caching expensive physics
    augmentations.
    """

    def __init__(self, iterator: Iterator[Mapping[str, torch.Tensor]]):
        self._items: List[Mapping[str, torch.Tensor]] = list(iterator)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._items)

    def __getitem__(self, idx: int) -> Mapping[str, torch.Tensor]:
        return self._items[idx]


class MAESyntheticOverheadDataset(Dataset[Mapping[str, torch.Tensor]]):
    """Generate synthetic overhead cubes suitable for masked autoencoding.

    The dataset produces flattened tube tokens extracted from a randomly
    generated hyperspectral cube. Randomness is fully deterministic per index
    when ``seed`` is provided.
    """

    def __init__(
        self,
        *,
        length: int,
        channels: int,
        image_size: int,
        patch_config: PatchConfig,
        seed: int = 0,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if length <= 0:
            raise ValueError("length must be positive")
        self.length = length
        self.channels = channels
        self.image_size = image_size
        self.patch_config = patch_config
        self._seed = int(seed)
        self._patch_sampler = PatchSampler(patch_config, random_offset=False)
        self.token_dim = channels * patch_config.height * patch_config.width

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.length

    def _rng(self, idx: int) -> Generator:
        g = torch.Generator()
        g.manual_seed(self._seed + idx)
        return g

    def _generate_cube(self, idx: int) -> Tensor:
        gen = self._rng(idx)
        cube = torch.randn(
            (self.channels, self.image_size, self.image_size), generator=gen
        )
        # Smooth the field to create spatial structure.
        cube = torch.nn.functional.avg_pool2d(
            cube.unsqueeze(0), kernel_size=3, stride=1, padding=1
        ).squeeze(0)
        cube = torch.nn.functional.avg_pool2d(
            cube.unsqueeze(0), kernel_size=3, stride=1, padding=1
        ).squeeze(0)
        return cube.to(dtype=torch.float32)

    def __getitem__(self, idx: int) -> Mapping[str, torch.Tensor]:
        cube = self._generate_cube(idx)
        tokens: List[Tensor] = []
        for patch in self._patch_sampler.sample(cube):
            tokens.append(patch.reshape(-1))
        if not tokens:
            raise RuntimeError("Patch configuration resulted in zero tokens")
        stacked = torch.stack(tokens, dim=0)
        return {
            "tokens": stacked,
            "target": stacked.clone(),
        }


def collate_mae(batch: List[Mapping[str, Tensor]]) -> Mapping[str, Tensor]:
    """Collate function that pads tokens to the maximum sequence length."""

    if not batch:
        raise ValueError("batch must not be empty")
    token_dim = batch[0]["tokens"].shape[-1]
    max_tokens = max(sample["tokens"].shape[0] for sample in batch)
    bsz = len(batch)

    tokens = torch.zeros(bsz, max_tokens, token_dim, dtype=torch.float32)
    targets = torch.zeros_like(tokens)
    mask = torch.zeros(bsz, max_tokens, dtype=torch.bool)

    for i, sample in enumerate(batch):
        current = sample["tokens"]
        length = current.shape[0]
        tokens[i, :length] = current
        targets[i, :length] = sample["target"]
        mask[i, :length] = True

    return {"tokens": tokens, "target": targets, "tokens_mask": mask}


class LabSpectraSynthetic(Dataset[Mapping[str, torch.Tensor]]):
    """Generate smooth synthetic lab reflectance spectra."""

    def __init__(
        self,
        *,
        length: int,
        hi_wvl_nm: Sequence[float] | Tensor,
        num_classes: int = 8,
        seed: int = 0,
    ) -> None:
        if length <= 0:
            raise ValueError("length must be positive")
        self.length = int(length)
        self.hi_wvl_nm = torch.as_tensor(hi_wvl_nm, dtype=torch.float32)
        if self.hi_wvl_nm.ndim != 1:
            raise ValueError("hi_wvl_nm must be 1-D")
        self.num_classes = max(1, int(num_classes))
        self._seed = int(seed)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.length

    def _rng(self, idx: int) -> Generator:
        g = torch.Generator()
        g.manual_seed(self._seed + idx)
        return g

    def __getitem__(self, idx: int) -> Mapping[str, torch.Tensor]:
        g = self._rng(idx)
        norm = (self.hi_wvl_nm - self.hi_wvl_nm.min())
        norm = norm / (norm.max() - norm.min() + 1e-6)
        freqs = torch.tensor([1.0, 2.0, 3.5], dtype=torch.float32)
        harmonics = torch.cos(freqs.unsqueeze(1) * norm.unsqueeze(0) * torch.pi)
        weights = 0.2 * torch.randn(len(freqs), generator=g)
        base = 0.55 + torch.sum(weights.unsqueeze(1) * harmonics, dim=0)
        trend_weight = 0.1 * torch.randn(1, generator=g)
        trend = torch.linspace(-0.5, 0.5, self.hi_wvl_nm.numel())
        spectrum = base + trend_weight * trend
        noise = 0.02 * torch.randn(self.hi_wvl_nm.shape[0], generator=g)
        spectrum = (spectrum + noise).clamp(0.02, 0.98).to(dtype=torch.float32)
        class_id = idx % self.num_classes
        return {
            "reflectance_hi": spectrum,
            "hi_wvl_nm": self.hi_wvl_nm,
            "class_id": torch.tensor(class_id, dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }


class OverheadSynthetic(Dataset[Mapping[str, torch.Tensor]]):
    """Generate synthetic overhead measurements via the renderer."""

    def __init__(
        self,
        *,
        length: int,
        lab_dataset: LabSpectraSynthetic,
        renderer: "LabToSensorRenderer",
        seed: int = 0,
        tau_jitter: float = 0.02,
        scale_jitter: float = 0.02,
        noise_std: float = 0.002,
    ) -> None:
        if length <= 0:
            raise ValueError("length must be positive")
        self.length = int(length)
        self.lab_dataset = lab_dataset
        self.renderer = renderer
        self._seed = int(seed)
        self.tau_base = float(renderer.rt_params.get("tau", 0.9))
        self.scale_base = float(renderer.rt_params.get("scale", 1.0))
        self.noise_base = float(renderer.rt_params.get("noise_std", 0.0))
        self.tau_jitter = float(tau_jitter)
        self.scale_jitter = float(scale_jitter)
        self.noise_jitter = float(noise_std)
        self.sensor_centers_nm = renderer.default_centers.clone().detach()
        self.sensor_fwhm_nm = renderer.default_fwhm.clone().detach()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.length

    def _rng(self, idx: int) -> Generator:
        g = torch.Generator()
        g.manual_seed(self._seed + idx)
        return g

    def _sample_rt(self, idx: int) -> Dict[str, float]:
        g = self._rng(idx)
        tau = self.tau_base + self.tau_jitter * torch.randn(1, generator=g).item()
        scale = self.scale_base + self.scale_jitter * torch.randn(1, generator=g).item()
        noise = abs(self.noise_base + self.noise_jitter * torch.randn(1, generator=g).item())
        return {"tau": tau, "scale": scale, "noise_std": noise}

    def __getitem__(self, idx: int) -> Mapping[str, torch.Tensor]:
        lab_idx = idx % len(self.lab_dataset)
        lab_sample = self.lab_dataset[lab_idx]
        reflectance = lab_sample["reflectance_hi"].unsqueeze(0)
        overrides = self._sample_rt(idx)
        sensor, _ = self.renderer.render(reflectance, rt_overrides=overrides)
        return {
            "radiance_sensor": sensor.squeeze(0),
            "lab_index": torch.tensor(lab_idx, dtype=torch.long),
            "sensor_centers_nm": self.sensor_centers_nm,
            "sensor_fwhm_nm": self.sensor_fwhm_nm,
            "rt_params": overrides,
        }


def collate_lab(batch: Sequence[Mapping[str, torch.Tensor]]) -> Mapping[str, torch.Tensor]:
    if not batch:
        raise ValueError("batch must not be empty")
    hi = batch[0]["hi_wvl_nm"].to(dtype=torch.float32)
    reflectance = torch.stack([sample["reflectance_hi"] for sample in batch], dim=0)
    class_id = torch.stack([sample["class_id"] for sample in batch], dim=0)
    indices = torch.stack([sample["index"] for sample in batch], dim=0)
    return {
        "reflectance_hi": reflectance,
        "hi_wvl_nm": hi,
        "class_id": class_id,
        "index": indices,
    }


def collate_overhead(batch: Sequence[Mapping[str, torch.Tensor]]) -> Mapping[str, torch.Tensor]:
    if not batch:
        raise ValueError("batch must not be empty")
    radiance = torch.stack([sample["radiance_sensor"] for sample in batch], dim=0)
    lab_indices = torch.stack([sample["lab_index"] for sample in batch], dim=0)
    centers = torch.stack(
        [sample["sensor_centers_nm"].to(dtype=torch.float32) for sample in batch], dim=0
    )
    fwhm = torch.stack(
        [sample["sensor_fwhm_nm"].to(dtype=torch.float32) for sample in batch], dim=0
    )
    keys = list(batch[0]["rt_params"].keys())
    rt_params = {
        key: torch.tensor([sample["rt_params"][key] for sample in batch], dtype=torch.float32)
        for key in keys
    }
    return {
        "radiance_sensor": radiance,
        "lab_index": lab_indices,
        "sensor_centers_nm": centers,
        "sensor_fwhm_nm": fwhm,
        "rt_params": rt_params,
    }


__all__ = [
    "SceneMetadata",
    "HyperspectralCube",
    "LabSpectraDataset",
    "CachedIterableDataset",
    "MAESyntheticOverheadDataset",
    "LabSpectraSynthetic",
    "OverheadSynthetic",
    "collate_mae",
    "collate_lab",
    "collate_overhead",
]
