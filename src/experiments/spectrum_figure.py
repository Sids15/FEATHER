"""Plot the Fisher eigenvalue spectrum from monitoring fit.json files.

Reads every fit.json under --fits-dir (any layout; files may be renamed
*_fit.json), groups them by mode, and draws the top-of-spectrum eigenvalues
on a log axis. The point of the figure is the cliff at index C-1 = 9: nine
informative eigenvalues, then a drop of ~14 orders of magnitude to numerical
zero, exactly as the rank bound predicts.

Output: paper/figures/fisher_spectrum.png.
Run: python src/experiments/spectrum_figure.py --fits-dir <folder>
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("feather.experiments.spectrum")

# Palette shared with analyze_monitoring.py (dataviz reference, light mode).
BLUE, VIOLET, RED, MUTED, INK = "#2a78d6", "#4a3aa7", "#e34948", "#898781", "#0b0b0b"
GRID, BASELINE, SURFACE, SECONDARY = "#e1e0d9", "#c3c2b7", "#fcfcfb", "#52514e"

TITLES = {
    "cifar10c": "ResNet-18 / CIFAR-10 ($d=512$)",
    "rotated_mnist": "CNN / MNIST ($d=128$)",
}


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits-dir", type=Path, required=True,
                        help="folder containing fit.json files (searched recursively)")
    parser.add_argument("--out", type=Path,
                        default=Path("paper/figures/fisher_spectrum.png"))
    args = parser.parse_args()

    spectra: dict[str, list[list[float]]] = defaultdict(list)
    for path in sorted(args.fits_dir.rglob("*fit.json")):
        fit = json.loads(path.read_text())
        spectra[fit["mode"]].append(fit["fisher_top_eigenvalues"])
        logger.info("loaded %s (%s)", path.name, fit["mode"])
    if not spectra:
        raise SystemExit(f"no fit.json files under {args.fits_dir}")

    fig, axes = plt.subplots(1, len(spectra), figsize=(8.4, 3.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes)
    for ax, mode in zip(axes, ("cifar10c", "rotated_mnist")):
        style_axis(ax)
        runs = spectra[mode]
        indices = np.arange(1, len(runs[0]) + 1)
        for eigenvalues in runs:
            ax.plot(indices, eigenvalues, color=VIOLET, alpha=0.35, linewidth=1.0)
        mean_spec = np.mean(runs, axis=0)
        ax.plot(indices, mean_spec, color=VIOLET, linewidth=2.0,
                marker="o", markersize=3.5)
        ax.set_yscale("log")
        ax.axvline(9.5, color=RED, linewidth=1.0, linestyle="--")
        ax.annotate("$C-1=9$", xy=(9.5, mean_spec[0]), xytext=(6.2, mean_spec[0]),
                    color=RED, fontsize=8, ha="right", va="center")
        ax.set_xticks(indices)
        ax.set_xlabel("eigenvalue index", color=SECONDARY, fontsize=9)
        ax.set_title(TITLES.get(mode, mode), color=INK, fontsize=10)
    axes[0].set_ylabel(r"eigenvalue $\lambda_j$ (log scale)",
                       color=SECONDARY, fontsize=9)
    fig.suptitle("Empirical Fisher spectra, five seeds per model",
                 color=INK, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor=SURFACE)
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
