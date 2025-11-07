from __future__ import annotations

from pathlib import Path

import tomllib
from setuptools import setup


def _load_metadata() -> dict[str, object]:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    metadata: dict[str, object] = {
        "name": project["name"],
        "version": project["version"],
        "description": project.get("description", ""),
        "long_description": Path(project.get("readme", "README.md")).read_text(encoding="utf-8"),
        "long_description_content_type": "text/markdown",
        "python_requires": project.get("requires-python", ">=3.11"),
        "install_requires": project.get("dependencies", []),
        "extras_require": project.get("optional-dependencies", {}),
        "packages": ["hsi_fm"],
        "package_dir": {"hsi_fm": "hsi_fm"},
    }
    authors = project.get("authors", [])
    if authors:
        first = authors[0]
        metadata["author"] = first.get("name", "")
        metadata["author_email"] = first.get("email", "")
    license_cfg = project.get("license")
    if isinstance(license_cfg, dict):
        if "file" in license_cfg:
            metadata["license_files"] = [license_cfg["file"]]
        elif "text" in license_cfg:
            metadata["license"] = license_cfg["text"]
    classifiers = project.get("classifiers", [])
    if classifiers:
        metadata["classifiers"] = classifiers
    scripts = project.get("scripts", {})
    if scripts:
        metadata["entry_points"] = {
            "console_scripts": [f"{name}={target}" for name, target in scripts.items()]
        }
    return metadata


setup(**_load_metadata())
