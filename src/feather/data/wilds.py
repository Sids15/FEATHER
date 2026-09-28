"""WILDS Camelyon17 loaders for the natural-shift monitoring experiment.

Camelyon17 is a binary tumor-vs-normal patch task (3x96x96) whose domains are
hospitals. The in-distribution ``train`` split (hospitals 0/3/4) is the clean
reference the monitor is fitted on; ``id_val`` is the in-distribution clean
episode; ``val`` (hospital 1) and ``test`` (hospital 2) are the natural
out-of-distribution drift episodes, harm graded by measured accuracy drop.

The ``wilds`` package is imported lazily and the dataset is expected to be
downloaded already (``python -c "from wilds import get_dataset;
get_dataset('camelyon17', root_dir=..., download=True)"`` once, ~10 GB), matching
the manual-download convention of the other loaders (docs/datasets.md).
Being binary, rank(F) <= C-1 = 1, so the blind subspace is nearly the whole
512-d feature space --- a clean structural demonstration on real data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision import transforms

# Camelyon17 patches are stain-normalized RGB; ImageNet statistics are the
# standard choice and match an ImageNet-initialised backbone.
CAMELYON17_MEAN = (0.485, 0.456, 0.406)
CAMELYON17_STD = (0.229, 0.224, 0.225)


def _transform() -> transforms.Compose:
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CAMELYON17_MEAN, CAMELYON17_STD)]
    )


class _StripMetadata(Dataset):
    """Wrap a WILDS subset so ``__getitem__`` yields (image, int label) only."""

    def __init__(self, wilds_subset) -> None:
        self._subset = wilds_subset

    def __len__(self) -> int:
        return len(self._subset)

    def __getitem__(self, index: int):
        x, y, _metadata = self._subset[index]
        return x, int(y)


def _get_dataset(data_root: str | Path | None):
    try:
        from wilds import get_dataset
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "the 'wilds' package is required for the Camelyon17 experiment; "
            "install it with 'pip install wilds' and download the data once via "
            "get_dataset('camelyon17', root_dir=..., download=True)"
        ) from exc
    root = str(Path(data_root)) if data_root is not None else None
    return get_dataset(dataset="camelyon17", root_dir=root, download=False)


def _subset(wilds_subset, max_samples: int | None, seed: int) -> Dataset:
    stripped = _StripMetadata(wilds_subset)
    if max_samples is None or max_samples >= len(stripped):
        return stripped
    idx = np.random.default_rng(seed).permutation(len(stripped))[:max_samples].tolist()
    return Subset(stripped, idx)


def camelyon17_split(
    data_root: str | Path | None = None,
    split: str = "train",
    max_samples: int | None = None,
    seed: int = 0,
) -> Dataset:
    """Return one Camelyon17 split (``train``/``id_val``/``val``/``test``)."""
    dataset = _get_dataset(data_root)
    return _subset(dataset.get_subset(split, transform=_transform()), max_samples, seed)


def camelyon17_reference(
    data_root: str | Path | None = None,
    max_samples: int | None = 30_000,
    seed: int = 0,
) -> Dataset:
    """In-distribution ``train`` split, optionally subsampled for the Fisher fit."""
    return camelyon17_split(data_root, "train", max_samples, seed)


def camelyon17_episodes(
    data_root: str | Path | None = None,
    max_samples: int | None = 20_000,
    seed: int = 0,
):
    """Yield (episode_name, dataset): clean (id_val) first, then OOD hospitals.

    ``max_samples`` caps each episode's length so the streams stay comparable;
    pass ``None`` to stream the full splits.
    """
    dataset = _get_dataset(data_root)
    for name, split in (("clean", "id_val"), ("ood_val_hosp1", "val"),
                        ("ood_test_hosp2", "test")):
        yield name, _subset(dataset.get_subset(split, transform=_transform()),
                            max_samples, seed)
