"""Dataset path resolution.

The data root is resolved, in priority order, from (1) an explicit override
argument, (2) the ``FEATHER_DATA_DIR`` environment variable, (3) ``./data``.
Code never downloads datasets (rules.md §3) — every missing-path error points
to docs/datasets.md instead.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ENV = "FEATHER_DATA_DIR"
_HINT = (
    f"Set the {DATA_ENV} environment variable (or pass --data-root) to the "
    "folder holding the datasets, and see docs/datasets.md for the required "
    "layout and official download links."
)


def data_root(override: str | Path | None = None) -> Path:
    """Return the dataset root directory (not required to exist yet)."""
    if override is not None:
        return Path(override)
    env = os.environ.get(DATA_ENV)
    if env:
        return Path(env)
    return Path("data")


def require_dir(path: Path, what: str) -> Path:
    """Return ``path`` if it is an existing directory, else raise with help."""
    if not path.is_dir():
        raise FileNotFoundError(f"{what} not found at '{path}'. {_HINT}")
    return path


def require_file(path: Path, what: str) -> Path:
    """Return ``path`` if it is an existing file, else raise with help."""
    if not path.is_file():
        raise FileNotFoundError(f"{what} not found at '{path}'. {_HINT}")
    return path


def mnist_root(override: str | Path | None = None) -> Path:
    """Root passed to torchvision MNIST; validates data/MNIST/raw exists."""
    root = data_root(override)
    raw = root / "MNIST" / "raw"
    require_dir(raw, "MNIST raw folder")
    probes = ("train-images-idx3-ubyte", "train-images-idx3-ubyte.gz")
    if not any((raw / name).is_file() for name in probes):
        raise FileNotFoundError(f"MNIST idx files not found in '{raw}'. {_HINT}")
    return root


def cifar10_root(override: str | Path | None = None) -> Path:
    """Root passed to torchvision CIFAR10; accepts the tar.gz or extracted dir."""
    root = data_root(override)
    if not (root / "cifar-10-python.tar.gz").is_file() and not (
        root / "cifar-10-batches-py"
    ).is_dir():
        raise FileNotFoundError(
            f"CIFAR-10 (cifar-10-python.tar.gz) not found under '{root}'. {_HINT}"
        )
    return root


def cifar10c_dir(override: str | Path | None = None) -> Path:
    """Directory holding the extracted CIFAR-10-C .npy files."""
    root = data_root(override)
    directory = require_dir(root / "CIFAR-10-C", "CIFAR-10-C folder")
    require_file(directory / "labels.npy", "CIFAR-10-C labels.npy")
    return directory
