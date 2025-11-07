"""Top-level package for the hyperspectral foundation model."""
from importlib import metadata

__all__ = ["__version__"]


def _get_version() -> str:
    try:
        return metadata.version("hsi-fm")
    except metadata.PackageNotFoundError:  # pragma: no cover - fallback for dev installs
        return "0.0.0"


__version__ = _get_version()
