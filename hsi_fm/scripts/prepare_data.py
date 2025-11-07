"""Data preparation utilities."""
from __future__ import annotations

import argparse
from pathlib import Path


def prepare_data(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.txt").write_text("Synthetic data placeholder\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare hyperspectral datasets")
    parser.add_argument("output", type=Path, help="Output directory")
    args = parser.parse_args()
    prepare_data(args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
