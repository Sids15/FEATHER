"""BREEDS-style subpopulation shift on CIFAR-100 (real labels, natural images).

The model is trained on CIFAR-100's 20 superclasses using only 4 of the 5
subclasses in each; the held-out 5th subclass is the deployment shift. Every
deployment image keeps its real superclass label, so any accuracy drop is
measured, never constructed. This is the shift that can be harmful while
output-invisible: *different* inputs, with their own true labels, that the
frozen head may map to the same output distribution.

Episodes (all 8000 images, balanced across superclasses unless noted):

- ``clean``: the seen-subclass test images (the in-distribution stream);
- ``mix_RRR``: a fraction rho = RRR/100 of each batch drawn from held-out
  subclasses, the rest from the seen test pool (the gradual-mixing sweep);
- ``swap_<superclass>``: one superclass's seen test images replaced by its
  held-out subclass, every other superclass clean (per-superclass harm).

The split/mixing logic is NumPy-only (testable without torch); torch is
imported only by the dataset builders.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from feather.data.paths import cifar100_dir

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)
N_SUPERCLASSES = 20
SUBCLASSES_PER_SUPERCLASS = 5
DEFAULT_RHOS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)


def load_cifar100(split: str, data_root: str | Path | None = None):
    """Return (uint8 images NHWC, fine labels, coarse labels) for 'train'/'test'."""
    # Pickle is the official CIFAR-100 distribution format (torchvision's own
    # CIFAR100 unpickles these same files); read directly only because
    # torchvision drops the coarse labels. Load only the manually downloaded
    # official archive (docs/datasets.md), never files from untrusted sources.
    with (cifar100_dir(data_root) / split).open("rb") as f:
        raw = pickle.load(f, encoding="bytes")
    images = raw[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images, np.asarray(raw[b"fine_labels"]), np.asarray(raw[b"coarse_labels"])


def superclass_names(data_root: str | Path | None = None) -> list[str]:
    # same official-archive pickle as load_cifar100
    with (cifar100_dir(data_root) / "meta").open("rb") as f:
        meta = pickle.load(f, encoding="bytes")
    return [n.decode() for n in meta[b"coarse_label_names"]]


def split_subclasses(fine: np.ndarray, coarse: np.ndarray,
                     holdout_rank: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Return (seen fine ids, held-out fine ids): per superclass, its fine ids
    sorted ascending, the one at ``holdout_rank`` held out (a fixed,
    pre-declared rule, not chosen by looking at results)."""
    seen, held = [], []
    for c in range(N_SUPERCLASSES):
        subs = np.unique(fine[coarse == c])
        if len(subs) != SUBCLASSES_PER_SUPERCLASS:
            raise ValueError(f"superclass {c} has {len(subs)} subclasses, expected 5")
        held.append(subs[holdout_rank])
        seen.extend(np.delete(subs, holdout_rank))
    return np.array(sorted(seen)), np.array(sorted(held))


def mix_indices(n_clean: int, n_shift: int, size: int, rho: float,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Indices for one mixed episode: ``round(rho*size)`` shifted samples and the
    rest clean, drawn without replacement and shuffled together so every stream
    batch carries about the same fraction rho. Returns (source, index): source
    0 = clean pool, 1 = shift pool."""
    k = int(round(rho * size))
    if size - k > n_clean or k > n_shift:
        raise ValueError(f"pools too small for size={size}, rho={rho}")
    source = np.r_[np.zeros(size - k, int), np.ones(k, int)]
    index = np.r_[rng.choice(n_clean, size - k, replace=False),
                  rng.choice(n_shift, k, replace=False)]
    order = rng.permutation(size)
    return source[order], index[order]


def swap_indices(clean_coarse: np.ndarray, shift_coarse: np.ndarray, superclass: int,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Indices for a swap episode: every clean image except ``superclass``'s,
    plus as many held-out images of ``superclass`` as it had clean ones, so the
    class mix (and with it the output histogram) is unchanged."""
    keep = np.flatnonzero(clean_coarse != superclass)
    n_swap = int((clean_coarse == superclass).sum())
    pool = np.flatnonzero(shift_coarse == superclass)
    take = rng.choice(pool, n_swap, replace=False)
    source = np.r_[np.zeros(len(keep), int), np.ones(n_swap, int)]
    index = np.r_[keep, take]
    order = rng.permutation(len(source))
    return source[order], index[order]


def gather(source: np.ndarray, index: np.ndarray, clean: tuple, shift: tuple):
    """Materialize (images, labels) for an episode from its (source, index)."""
    x = np.empty((len(source),) + clean[0].shape[1:], clean[0].dtype)
    y = np.empty(len(source), np.int64)
    for s, (px, py) in ((0, clean), (1, shift)):
        m = source == s
        x[m], y[m] = px[index[m]], py[index[m]]
    return x, y


# ---------------------------------------------------------------- torch side

def _to_tensor_dataset(images: np.ndarray, labels: np.ndarray):
    import torch
    from torch.utils.data import TensorDataset

    x = torch.from_numpy(np.ascontiguousarray(images)).float().permute(0, 3, 1, 2) / 255.0
    x = (x - torch.tensor(CIFAR100_MEAN).view(1, 3, 1, 1)) / torch.tensor(CIFAR100_STD).view(1, 3, 1, 1)
    return TensorDataset(x, torch.from_numpy(labels.astype(np.int64)))


class _AugmentedArrays:
    """Train-time dataset with the standard CIFAR crop-and-flip augmentation."""

    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        from torchvision import transforms

        self.images, self.labels = images, labels.astype(np.int64)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i):
        return self.transform(self.images[i]), int(self.labels[i])


def _pools(data_root, holdout_rank):
    tr_x, tr_f, tr_c = load_cifar100("train", data_root)
    te_x, te_f, te_c = load_cifar100("test", data_root)
    seen, held = split_subclasses(tr_f, tr_c, holdout_rank)
    seen_tr, seen_te = np.isin(tr_f, seen), np.isin(te_f, seen)
    # held-out subclasses were never trained on, so both splits form the pool
    shift_x = np.concatenate([tr_x[~seen_tr], te_x[~seen_te]])
    shift_c = np.concatenate([tr_c[~seen_tr], te_c[~seen_te]])
    return (tr_x[seen_tr], tr_c[seen_tr]), (te_x[seen_te], te_c[seen_te]), (shift_x, shift_c)


def breeds_datasets(data_root=None, augment: bool = True, holdout_rank: int = 4):
    """(train, val) over the seen subclasses, labelled by superclass."""
    (tr_x, tr_c), (te_x, te_c), _ = _pools(data_root, holdout_rank)
    train = _AugmentedArrays(tr_x, tr_c) if augment else _to_tensor_dataset(tr_x, tr_c)
    return train, _to_tensor_dataset(te_x, te_c)


def breeds_episodes(data_root=None, rhos=DEFAULT_RHOS, seed: int = 0, holdout_rank: int = 4):
    """Yield (episode name, dataset): clean, the rho-mixing sweep, then swaps."""
    _, (cl_x, cl_c), (sh_x, sh_c) = _pools(data_root, holdout_rank)
    rng = np.random.default_rng(seed)

    def build(source, index):
        return _to_tensor_dataset(*gather(source, index, (cl_x, cl_c), (sh_x, sh_c)))

    yield "clean", _to_tensor_dataset(cl_x, cl_c)
    for rho in rhos:
        if rho > 0:
            yield f"mix_{int(round(rho * 100)):03d}", build(*mix_indices(len(cl_c), len(sh_c), len(cl_c), rho, rng))
    for c, name in enumerate(superclass_names(data_root)):
        yield f"swap_{name}", build(*swap_indices(cl_c, sh_c, c, rng))
