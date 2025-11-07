from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from hsi_fm.data.overhead import OverheadHSI, collate_overhead
from hsi_fm.data.patches import PatchConfig


class _StubDataset:
    def __init__(self) -> None:
        self.variables: Dict[str, Any] = {
            "reflectance": np.ones((3, 4, 4), dtype=np.float32),
            "qa_mask": np.ones((4, 4), dtype=np.int8),
            "wavelengths": np.array([500.0, 600.0, 700.0], dtype=np.float32),
            "srf": np.ones((3, 3), dtype=np.float32),
        }
        self.solar_zenith = np.array([30.0])

    def __enter__(self) -> "_StubDataset":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _fake_loader(path: Path) -> _StubDataset:  # pragma: no cover - test helper
    return _StubDataset()


def test_overhead_dataset_tiles_and_collates(tmp_path: Path) -> None:
    dummy = tmp_path / "dummy.npz"
    np.savez(dummy, reflectance=np.ones((3, 4, 4)))
    dataset = OverheadHSI(
        source="emit_l2a",
        paths=[dummy],
        patch_config=PatchConfig(height=2, width=2, stride=2),
        loader=_fake_loader,
        min_valid_fraction=0.1,
    )
    assert len(dataset) == 4
    sample = dataset[0]
    assert sample["tokens"].shape == (4, 3)
    batch = collate_overhead([dataset[0], dataset[1]])
    assert batch["tokens"].shape == (2, 4, 3)
    assert batch["qa_mask"].shape[-2:] == (2, 2)
    assert torch.all(batch["wavelengths"] > 0)
