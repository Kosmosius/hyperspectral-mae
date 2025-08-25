from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.hashing import sha256_path, stable_meta_hash


class ManifestError(ValueError):
    """Raised when a manifest is missing, malformed, or fails validation."""


@dataclass(frozen=True)
class BaseManifest:
    name: str
    version: str
    license: str
    source: str
    citation_keys: Tuple[str, ...]
    preprocessing: Dict[str, Any]
    sha256: Dict[str, str]
    notes: str

    @classmethod
    def from_json(cls, obj: Dict[str, Any]) -> "BaseManifest":
        try:
            return cls(
                name=str(obj["name"]),
                version=str(obj["version"]),
                license=str(obj.get("license", "")),
                source=str(obj.get("source", "")),
                citation_keys=tuple(obj.get("citation_keys", [])),
                preprocessing=dict(obj.get("preprocessing", {})),
                sha256=dict(obj.get("sha256", {})),
                notes=str(obj.get("notes", "")),
            )
        except Exception as e:
            raise ManifestError(f"Base manifest parse error: {e}") from e

    def minimal_validate(self) -> None:
        if not self.name or not self.version:
            raise ManifestError("Manifest requires 'name' and 'version'")
        # Ensure sha256 is mapping of filename->hex
        for k, v in self.sha256.items():
            if not isinstance(k, str) or not isinstance(v, str) or len(v) < 32:
                raise ManifestError("sha256 map must be {filename: hex_digest}")


@dataclass(frozen=True)
class LabManifest(BaseManifest):
    # Expected files
    canonical_grid_csv: str                   # e.g., canonical/Lambda_c_400_2500nm_3nm.csv
    spectra_npz: str                          # e.g., lab/usgs_v7/canonical/spectra.npz
    label_map_csv: Optional[str] = None       # optional mapping of spectrum_id -> family

    @classmethod
    def from_json(cls, obj: Dict[str, Any]) -> "LabManifest":
        base = BaseManifest.from_json(obj)
        lab = obj.get("lab", {})
        if "canonical_grid_csv" not in lab or "spectra_npz" not in lab:
            raise ManifestError("Lab manifest requires 'lab.canonical_grid_csv' and 'lab.spectra_npz'")
        return cls(
            **base.__dict__,
            canonical_grid_csv=str(lab["canonical_grid_csv"]),
            spectra_npz=str(lab["spectra_npz"]),
            label_map_csv=str(lab.get("label_map_csv")) if lab.get("label_map_csv") else None,
        )


@dataclass(frozen=True)
class FieldManifest(BaseManifest):
    canonical_grid_csv: str                   # canonical grid path
    spectra_npz: str                          # field spectra on canonical grid
    metadata_csv: Optional[str] = None        # optional per-record metadata

    @classmethod
    def from_json(cls, obj: Dict[str, Any]) -> "FieldManifest":
        base = BaseManifest.from_json(obj)
        fld = obj.get("field", {})
        if "canonical_grid_csv" not in fld or "spectra_npz" not in fld:
            raise ManifestError("Field manifest requires 'field.canonical_grid_csv' and 'field.spectra_npz'")
        return cls(
            **base.__dict__,
            canonical_grid_csv=str(fld["canonical_grid_csv"]),
            spectra_npz=str(fld["spectra_npz"]),
            metadata_csv=str(fld.get("metadata_csv")) if fld.get("metadata_csv") else None,
        )


@dataclass(frozen=True)
class SatelliteManifest(BaseManifest):
    canonical_grid_csv: str                   # canonical grid path (matches R)
    cube_npz: str                             # e.g., satellite/PRISMA/scene_xxx/cube.npz (y, masks, etc.)
    srf_matrix_npz: str                       # mission SRF matrix on CanonicalGrid (R)
    masks_npz: Optional[str] = None           # optional extra masks
    mission: Optional[str] = None             # e.g., "PRISMA" | "EnMAP"

    @classmethod
    def from_json(cls, obj: Dict[str, Any]) -> "SatelliteManifest":
        base = BaseManifest.from_json(obj)
        sat = obj.get("satellite", {})
        required = ("canonical_grid_csv", "cube_npz", "srf_matrix_npz")
        if not all(k in sat for k in required):
            raise ManifestError("Satellite manifest requires 'satellite.{canonical_grid_csv,cube_npz,srf_matrix_npz}'")
        return cls(
            **base.__dict__,
            canonical_grid_csv=str(sat["canonical_grid_csv"]),
            cube_npz=str(sat["cube_npz"]),
            srf_matrix_npz=str(sat["srf_matrix_npz"]),
            masks_npz=str(sat.get("masks_npz")) if sat.get("masks_npz") else None,
            mission=str(sat.get("mission")) if sat.get("mission") else None,
        )


def _load_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise ManifestError(f"Manifest not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest(path: str | Path) -> BaseManifest:
    obj = _load_json(path)
    dtype = obj.get("datatype", "")
    if dtype == "lab":
        return LabManifest.from_json(obj)
    elif dtype == "field":
        return FieldManifest.from_json(obj)
    elif dtype == "satellite":
        return SatelliteManifest.from_json(obj)
    else:
        # Fallback: try to infer based on nested keys
        if "lab" in obj:
            return LabManifest.from_json(obj)
        if "field" in obj:
            return FieldManifest.from_json(obj)
        if "satellite" in obj:
            return SatelliteManifest.from_json(obj)
        raise ManifestError("Unknown manifest type; expected 'datatype' or 'lab/field/satellite' section")


def validate_manifest_file(path: str | Path, verify_hashes: bool = True, root: Optional[str | Path] = None) -> BaseManifest:
    """
    Load and minimally validate a manifest. Optionally verify file hashes listed in sha256.
    """
    m = load_manifest(path)
    m.minimal_validate()

    if verify_hashes:
        base_dir = Path(root) if root else Path(path).parent
        missing: List[str] = []
        mismatched: List[Tuple[str, str, str]] = []  # (file, expected, actual)
        for file_rel, expected in m.sha256.items():
            fp = base_dir / file_rel
            if not fp.is_file():
                missing.append(str(fp))
                continue
            actual = sha256_path(fp)
            if actual != expected:
                mismatched.append((str(fp), expected, actual))
        if missing:
            raise ManifestError(f"Missing files referenced by manifest: {missing}")
        if mismatched:
            msg = "; ".join(f"{f} expected={e} actual={a}" for f, e, a in mismatched)
            raise ManifestError(f"Hash mismatch for files: {msg}")

    return m
