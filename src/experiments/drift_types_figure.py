"""Render the textbook drift-pattern illustration for the conference paper.

Four panels showing how a data statistic can evolve over time: sudden,
gradual, incremental, and recurring drift. Entirely schematic (no project
data), matching the taxonomy discussed in Gama et al. (2014) and
Lu et al. (2019).

Output: paper-conference/figures/drift_types.png.
Run: python src/experiments/drift_types_figure.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("feather.experiments.drift_types")

# Palette shared with the other figures (dataviz reference, light mode).
BLUE, VIOLET, RED, MUTED, INK = "#2a78d6", "#4a3aa7", "#e34948", "#898781", "#0b0b0b"
GRID, BASELINE, SURFACE, SECONDARY = "#e1e0d9", "#c3c2b7", "#fcfcfb", "#52514e"


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(-0.35, 1.45)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("paper-conference/figures/drift_types.png"))
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 400)
    noise = lambda: rng.normal(0, 0.035, t.size)  # noqa: E731

    sudden = np.where(t < 0.5, 0.0, 1.0) + noise()
    ramp = np.clip((t - 0.35) / 0.3, 0, 1)
    # gradual: increasingly frequent switches to the new concept
    gradual = (rng.random(t.size) < ramp).astype(float) * 1.0
    gradual = np.convolve(gradual, np.ones(9) / 9, mode="same") + noise()
    incremental = ramp + noise()
    recurring = (np.sin(2 * np.pi * 1.5 * t) > 0).astype(float) + noise()

    panels = [
        ("sudden", sudden),
        ("gradual", gradual),
        ("incremental", incremental),
        ("recurring", recurring),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(9.0, 2.1), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    for ax, (name, series) in zip(axes, panels):
        style_axis(ax)
        ax.plot(t, series, color=VIOLET, linewidth=1.1)
        ax.axhline(0.0, color=GRID, linewidth=0.8, zorder=0)
        ax.axhline(1.0, color=GRID, linewidth=0.8, zorder=0)
        ax.set_title(name, color=INK, fontsize=10)
        ax.set_xlabel("time", color=SECONDARY, fontsize=8.5)
    axes[0].set_ylabel("data statistic", color=SECONDARY, fontsize=8.5)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor=SURFACE)
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
