"""Dataset wrappers for overhead hyperspectral data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from hsi_fm.data.emit import EMITSample, read_emit_l1b, read_emit_l2a
from hsi_fm.data.enmap import EnMAPSample, read_enmap_l2a
from hsi_fm.data.patches import PatchConfig, PatchSampler


ReaderFn = Callable[[Path], EMITSample | EnMAPSample]


@dataclass
class OverheadPatch:
    tokens: torch.Tensor
    target: torch.Tensor
    qa_mask: torch.BoolTensor
    wavelengths: torch.Tensor
    geometry: Dict[str, torch.Tensor]
    sensor: str
    srf: Optional[torch.Tensor]


class OverheadHSI(Dataset[Dict[str, torch.Tensor]]):
    """Tiles EMIT/EnMAP scenes into tube tokens."""

    def __init__(
        self,
        source: str,
        paths: Sequence[str | Path],
        patch_config: PatchConfig,
        *,
        loader: Optional[Callable[..., EMITSample | EnMAPSample]] = None,
        random_offset: bool = False,
        min_valid_fraction: float = 0.5,
        **reader_kwargs,
    ) -> None:
        self.source = source
        self.paths = [Path(p) for p in paths]
        self.patch_config = patch_config
        self._patches: List[OverheadPatch] = []
        if random_offset:
            raise ValueError("Random offsets are not supported to ensure QA mask alignment")
        sampler = PatchSampler(patch_config, random_offset=False)

        reader = self._select_reader(source, loader, reader_kwargs)

        mask_sampler = PatchSampler(patch_config, random_offset=False)
        for path in self.paths:
            sample = reader(path)
            mask_cube = sample.qa_mask.unsqueeze(0).float()
            for cube_patch, mask_patch in zip(sampler.sample(sample.cube), mask_sampler.sample(mask_cube)):
                qa_mask = mask_patch[0] > 0.5
                if qa_mask.numel() == 0:
                    continue
                if qa_mask.float().mean().item() < min_valid_fraction:
                    continue
                tokens = self._cube_to_tokens(cube_patch)
                wavelengths = sample.wavelengths_nm.clone()
                geometry = {
                    k: v.clone() if isinstance(v, torch.Tensor) else torch.as_tensor(v, dtype=torch.float32)
                    for k, v in sample.geometry.items()
                }
                self._patches.append(
                    OverheadPatch(
                        tokens=tokens,
                        target=tokens.clone(),
                        qa_mask=qa_mask,
                        wavelengths=wavelengths,
                        geometry=geometry,
                        sensor=sample.sensor_id,
                        srf=sample.srf.clone() if sample.srf is not None else None,
                    )
                )

    @staticmethod
    def _select_reader(
        source: str,
        loader: Optional[Callable[..., EMITSample | EnMAPSample]],
        reader_kwargs: Dict,
    ) -> ReaderFn:
        if source == "emit_l2a":
            return lambda path: read_emit_l2a(path, loader=loader, **reader_kwargs)
        if source == "emit_l1b":
            return lambda path: read_emit_l1b(path, loader=loader, **reader_kwargs)
        if source == "enmap_l2a":
            return lambda path: read_enmap_l2a(path, loader=loader, **reader_kwargs)
        raise ValueError(f"Unknown source '{source}'")

    @staticmethod
    def _cube_to_tokens(patch: torch.Tensor) -> torch.Tensor:
        if patch.ndim != 3:
            raise ValueError("Patch must be (C, H, W)")
        c, h, w = patch.shape
        return patch.reshape(c, h * w).transpose(0, 1)

    def __len__(self) -> int:
        return len(self._patches)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor] | str]:
        patch = self._patches[idx]
        return {
            "tokens": patch.tokens,
            "target": patch.target,
            "qa_mask": patch.qa_mask,
            "wavelengths": patch.wavelengths,
            "geometry": patch.geometry,
            "sensor": patch.sensor,
            "srf": patch.srf,
        }


def collate_overhead(batch: List[Dict[str, torch.Tensor | Dict[str, torch.Tensor] | str]]) -> Dict:
    tokens = torch.stack([item["tokens"] for item in batch])
    target = torch.stack([item["target"] for item in batch])
    qa_mask = torch.stack([item["qa_mask"] for item in batch])
    wavelengths = [item["wavelengths"] for item in batch]
    wavelengths_padded = pad_sequence(wavelengths, batch_first=True)
    wavelength_mask = pad_sequence(
        [torch.ones_like(w, dtype=torch.bool) for w in wavelengths], batch_first=True
    )
    srfs = [item.get("srf") for item in batch]
    srf_tensors = [s for s in srfs if s is not None]
    if srf_tensors:
        max_bands = max(s.shape[0] for s in srf_tensors)
        max_samples = max(s.shape[1] for s in srf_tensors)
        padded_srf = torch.zeros(len(batch), max_bands, max_samples)
        srf_mask = torch.zeros(len(batch), max_bands, max_samples, dtype=torch.bool)
        for idx, s in enumerate(srfs):
            if s is None:
                continue
            padded_srf[idx, : s.shape[0], : s.shape[1]] = s
            srf_mask[idx, : s.shape[0], : s.shape[1]] = True
    else:
        padded_srf = None
        srf_mask = None

    geometry_keys = sorted({k for item in batch for k in item["geometry"].keys()})
    geometry = {
        key: torch.stack(
            [
                item["geometry"].get(key, torch.tensor(float("nan"), dtype=torch.float32)).view(1)
                for item in batch
            ]
        ).squeeze(-1)
        for key in geometry_keys
    }

    return {
        "tokens": tokens,
        "target": target,
        "qa_mask": qa_mask,
        "wavelengths": wavelengths_padded,
        "wavelength_mask": wavelength_mask,
        "sensor": [item["sensor"] for item in batch],
        "geometry": geometry,
        "srf": padded_srf,
        "srf_mask": srf_mask,
    }


__all__ = ["OverheadHSI", "collate_overhead", "OverheadPatch"]

