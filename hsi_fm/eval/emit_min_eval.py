"""Minimal EMIT L2B solids evaluation CLI."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


Tensor = torch.Tensor


@dataclass
class MineralSample:
    mineral_ids: Tensor
    band_depths: Tensor
    reflectance: Tensor
    wavelengths: Tensor


def _load_npz(path: Path) -> MineralSample:
    data = np.load(path)
    return MineralSample(
        mineral_ids=torch.as_tensor(data["mineral_ids"], dtype=torch.long),
        band_depths=torch.as_tensor(data["band_depths"], dtype=torch.float32),
        reflectance=torch.as_tensor(data["reflectance"], dtype=torch.float32),
        wavelengths=torch.as_tensor(data["wavelengths"], dtype=torch.float32),
    )


def load_truth(glob_pattern: str) -> Dict[str, MineralSample]:
    result = {}
    for path in Path().glob(glob_pattern):
        result[path.stem] = _load_npz(path)
    return result


def load_predictions(directory: Path) -> Dict[str, MineralSample]:
    result = {}
    for file in directory.glob("*.npz"):
        result[file.stem] = _load_npz(file)
    return result


def spectral_angle(a: Tensor, b: Tensor) -> Tensor:
    dot = torch.sum(a * b, dim=-1)
    norm = torch.linalg.norm(a, dim=-1) * torch.linalg.norm(b, dim=-1)
    norm = norm.clamp_min(1e-8)
    angle = torch.arccos((dot / norm).clamp(-1.0, 1.0))
    return angle


def band_depth_mae(pred: Tensor, truth: Tensor) -> Tensor:
    return torch.mean(torch.abs(pred - truth), dim=-1)


def f1_per_class(pred: Tensor, truth: Tensor, num_classes: int) -> Tensor:
    scores = []
    for cls in range(num_classes):
        pred_pos = pred == cls
        truth_pos = truth == cls
        tp = torch.sum(pred_pos & truth_pos)
        fp = torch.sum(pred_pos & ~truth_pos)
        fn = torch.sum(~pred_pos & truth_pos)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        scores.append(f1)
    return torch.stack(scores)


def evaluate_pair(pred: MineralSample, truth: MineralSample) -> Dict[str, Tensor]:
    if pred.mineral_ids.shape != truth.mineral_ids.shape:
        raise ValueError("Prediction and truth mineral maps must align")
    num_classes = int(torch.max(torch.cat([pred.mineral_ids, truth.mineral_ids]))) + 1
    f1 = f1_per_class(pred.mineral_ids, truth.mineral_ids, num_classes)
    sam = spectral_angle(pred.reflectance, truth.reflectance)
    bd_mae = band_depth_mae(pred.band_depths, truth.band_depths)
    return {
        "f1": f1,
        "sam": sam.mean(),
        "band_depth_mae": bd_mae.mean(),
    }


def aggregate_metrics(metrics: Iterable[Dict[str, Tensor]]) -> Dict[str, float]:
    f1s: List[Tensor] = []
    sams: List[Tensor] = []
    bd: List[Tensor] = []
    for metric in metrics:
        f1s.append(metric["f1"])
        sams.append(metric["sam"])
        bd.append(metric["band_depth_mae"])
    macro_f1 = torch.stack([f.mean() for f in f1s]).mean().item() if f1s else 0.0
    per_class = torch.stack(f1s).mean(dim=0) if f1s else torch.zeros(1)
    sam_value = torch.stack(sams).mean().item() if sams else 0.0
    bd_value = torch.stack(bd).mean().item() if bd else 0.0
    return {
        "macro_f1": macro_f1,
        "per_class_f1": per_class.tolist(),
        "sam": sam_value,
        "band_depth_mae": bd_value,
    }


def evaluate_directory(pred_dir: Path, truth_glob: str) -> Dict[str, float]:
    truth = load_truth(truth_glob)
    predictions = load_predictions(pred_dir)
    metrics = []
    for key, truth_sample in truth.items():
        if key not in predictions:
            raise KeyError(f"Missing prediction for {key}")
        metrics.append(evaluate_pair(predictions[key], truth_sample))
    return aggregate_metrics(metrics)


def evaluate_emit_min(pred_dir: Path, truth_glob: str) -> Dict[str, float]:
    """Public API mirroring :func:`evaluate_directory`."""
    return evaluate_directory(pred_dir, truth_glob)


def write_csv(metrics: Dict[str, float], destination: Path) -> None:
    with destination.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EMIT L2B solids evaluation")
    parser.add_argument("--pred", required=True, type=Path, help="Directory with prediction npz files")
    parser.add_argument("--truth", required=True, help="Glob pattern for EMIT L2B truth npz files")
    parser.add_argument("--out", required=True, type=Path, help="Output CSV path")
    return parser


def main(args: List[str] | None = None) -> None:
    parser = build_parser()
    parsed = parser.parse_args(args)
    metrics = evaluate_directory(parsed.pred, parsed.truth)
    write_csv(metrics, parsed.out)


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "MineralSample",
    "evaluate_emit_min",
    "evaluate_directory",
    "load_predictions",
    "load_truth",
]

