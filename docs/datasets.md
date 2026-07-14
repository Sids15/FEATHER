# Datasets — Manual Download Guide

**Policy:** code in this repo never auto-downloads datasets (`download=False`
everywhere). Download the files below yourself and place them in the exact layout
shown. Everything is free; total ≈ 3.1 GB.

## Target layout

```
FEATHER/
└── data/                     ← git-ignored
    ├── cifar-10-python.tar.gz    (CIFAR-10, leave the tar.gz as-is)
    ├── CIFAR-10-C/               (extracted: 19 .npy corruption files + labels.npy)
    └── MNIST/raw/                (4 .gz files, leave gzipped)
```

## 1. CIFAR-10 (training data) — ~163 MB

- URL: <https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz>
- Put the file at `data/cifar-10-python.tar.gz` — **do not extract**; torchvision
  reads/extracts it itself with `root="data", download=False`.

## 2. CIFAR-10-C (corruption streams, main benchmark) — ~2.9 GB

- Official Zenodo record: <https://zenodo.org/records/2535967>
- Direct file link: <https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1>
- Download `CIFAR-10-C.tar` (ignore CIFAR-10-P unless we later want perturbation
  sequences), extract so the `.npy` files land in `data/CIFAR-10-C/`
  (e.g., `data/CIFAR-10-C/gaussian_noise.npy`, `data/CIFAR-10-C/labels.npy`).
- Mirror if Zenodo is slow: Hugging Face `randall-lab/cifar10-c`.

## 3. MNIST (for Rotated MNIST) — ~55 MB

Rotated MNIST has no canonical download — it is constructed on the fly from plain
MNIST + a rotation transform (standard practice). Download the 4 raw files from the
official PyTorch mirror (the original yann.lecun.com host often 403s):

- <https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz>
- <https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz>
- <https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz>
- <https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz>

Place them (still gzipped) in `data/MNIST/raw/`. One-paste PowerShell command
(run from the repo root):

```powershell
mkdir data\MNIST\raw -Force; foreach ($f in "train-images-idx3-ubyte.gz","train-labels-idx1-ubyte.gz","t10k-images-idx3-ubyte.gz","t10k-labels-idx1-ubyte.gz") { curl.exe -L -o "data\MNIST\raw\$f" "https://ossci-datasets.s3.amazonaws.com/mnist/$f" }
```

## 4. Tabular streams (Tier 4, optional — skip for now)

- **Electricity/Elec2**: bundled loader `river.datasets.Elec2` (~few MB; River
  fetches on first use — if we enable this tier we'll pre-fetch it once, manually).
- **Covertype**: <https://www.openml.org/d/1596> or
  `sklearn.datasets.fetch_covtype` (again, one manual pre-fetch).

## 5. Synthetic 2D (Tier 1)

Generated in-code with fixed seeds — nothing to download.
