"""Vision dataset loaders (MNIST, CIFAR-10, CIFAR-10-C).

All loaders use ``download=False`` and resolve paths via
:mod:`feather.data.paths` — datasets are downloaded manually per
docs/datasets.md. CIFAR-10-C files are the official Zenodo .npy arrays of
shape (50000, 32, 32, 3): five severities x 10000 images, severity ``s``
occupying rows ``[(s-1)*10000, s*10000)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, transforms

from feather.data.paths import cifar10_root, cifar10c_dir, mnist_root, require_file

MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

SEVERITIES = (1, 2, 3, 4, 5)
_IMAGES_PER_SEVERITY = 10_000

CIFAR10C_CORRUPTIONS = (
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog",
    "frost", "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise",
    "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise",
    "snow", "spatter", "speckle_noise", "zoom_blur",
)


def mnist_datasets(
    data_root: str | Path | None = None,
) -> tuple[Dataset, Dataset]:
    """Return (train, test) MNIST datasets with standard normalization."""
    root = mnist_root(data_root)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)]
    )
    train = datasets.MNIST(root, train=True, transform=transform, download=False)
    test = datasets.MNIST(root, train=False, transform=transform, download=False)
    return train, test


def cifar10_datasets(
    data_root: str | Path | None = None, augment: bool = True
) -> tuple[Dataset, Dataset]:
    """Return (train, test) CIFAR-10 datasets; train optionally augmented."""
    root = cifar10_root(data_root)
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        train_transform = test_transform
    train = datasets.CIFAR10(root, train=True, transform=train_transform, download=False)
    test = datasets.CIFAR10(root, train=False, transform=test_transform, download=False)
    return train, test


def load_cifar10c(
    corruption: str,
    severity: int,
    data_root: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one (corruption, severity) block of CIFAR-10-C as raw arrays.

    Args:
        corruption: One of :data:`CIFAR10C_CORRUPTIONS`.
        severity: Severity level in 1..5.
        data_root: Optional data-root override.

    Returns:
        (images, labels): uint8 images of shape (10000, 32, 32, 3) and int
        labels of shape (10000,).
    """
    if corruption not in CIFAR10C_CORRUPTIONS:
        raise ValueError(
            f"unknown corruption {corruption!r}; expected one of {CIFAR10C_CORRUPTIONS}"
        )
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be in {SEVERITIES}, got {severity}")
    directory = cifar10c_dir(data_root)
    images_path = require_file(
        directory / f"{corruption}.npy", f"CIFAR-10-C file {corruption}.npy"
    )
    start = (severity - 1) * _IMAGES_PER_SEVERITY
    stop = severity * _IMAGES_PER_SEVERITY
    images = np.load(images_path, mmap_mode="r")[start:stop]
    labels = np.load(directory / "labels.npy", mmap_mode="r")[start:stop]
    return np.asarray(images), np.asarray(labels).astype(np.int64)


def rotated_mnist_test(
    angle: float,
    data_root: str | Path | None = None,
) -> Dataset:
    """MNIST test set with every image rotated by ``angle`` degrees.

    The standard Rotated-MNIST drift construction: the rotation angle is the
    continuous drift-severity knob (0 = clean reference distribution).
    """
    root = mnist_root(data_root)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: transforms.functional.rotate(x, float(angle))
            ),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )
    return datasets.MNIST(root, train=False, transform=transform, download=False)


def cifar10c_dataset(
    corruption: str,
    severity: int,
    data_root: str | Path | None = None,
) -> TensorDataset:
    """CIFAR-10-C block as a normalized float TensorDataset (for streaming)."""
    images, labels = load_cifar10c(corruption, severity, data_root)
    x = torch.from_numpy(images).float().permute(0, 3, 1, 2) / 255.0
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
    x = (x - mean) / std
    return TensorDataset(x, torch.from_numpy(labels))
