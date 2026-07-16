"""Render the paper's two schematic diagrams as PNGs.

The Springer SNmult author instructions ask contributors to avoid TikZ
(it cannot be rendered in their non-PDF output formats), so the pipeline
overview and the blind-geometry cartoon are drawn here with matplotlib and
included in main.tex as ordinary images.

Outputs: paper/figures/pipeline.png, paper/figures/geometry_blind.png.
Run: python src/experiments/paper_diagrams.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

logger = logging.getLogger("feather.experiments.diagrams")

# Palette shared with the data figures (dataviz reference, light mode).
BLUE, VIOLET, RED, MUTED, INK = "#2a78d6", "#4a3aa7", "#e34948", "#898781", "#0b0b0b"
GRID, BASELINE, SURFACE, SECONDARY = "#e1e0d9", "#c3c2b7", "#fcfcfb", "#52514e"

BOX_KW = dict(boxstyle="round,pad=0.12", linewidth=0.9,
              edgecolor="#777777", facecolor="white")
ARROW_KW = dict(arrowstyle="-|>", mutation_scale=11, linewidth=1.0,
                color="#555555", shrinkA=4, shrinkB=4)


def box(ax: plt.Axes, x: float, y: float, text: str) -> tuple[float, float]:
    ax.add_patch(FancyBboxPatch((x - 0.62, y - 0.28), 1.24, 0.56, **BOX_KW))
    ax.text(x, y, text, ha="center", va="center", fontsize=8.2, color=INK)
    return x, y


def pipeline(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 2.9), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_xlim(-0.85, 5.6)
    ax.set_ylim(-0.65, 1.85)
    ax.axis("off")

    xs = [0.0, 1.55, 3.1, 4.65]
    top = [
        "labeled\nreference set",
        "activation Fisher\n$\\mathbf{F}\\in\\mathbb{R}^{d\\times d}$ (exact)",
        "eigendecomposition\n$\\mathcal{S}\\oplus\\mathcal{N}$",
        "bootstrap\nthresholds",
    ]
    bottom = [
        "stream batch\n$B_t$",
        "shift $\\Delta\\mu_t$,\nproject onto $\\mathcal{N}$",
        "statistics\n$s_t,\\ m_t,\\ v_t$",
        "alarm if over\ncalibrated threshold",
    ]
    for row, y in ((top, 1.25), (bottom, 0.0)):
        for label, x in zip(row, xs):
            box(ax, x, y, label)
        for x0, x1 in zip(xs[:-1], xs[1:]):
            ax.add_patch(FancyArrowPatch((x0 + 0.62, y), (x1 - 0.62, y),
                                         **ARROW_KW))
    # what the offline stage hands to the online stage
    dashed = dict(ARROW_KW, linestyle=(0, (4, 3)))
    ax.add_patch(FancyArrowPatch((xs[2], 0.95), (xs[1] + 0.35, 0.30),
                                 connectionstyle="arc3,rad=0.12", **dashed))
    ax.add_patch(FancyArrowPatch((xs[3], 0.95), (xs[3] + 0.35, 0.30),
                                 connectionstyle="arc3,rad=-0.12", **dashed))
    ax.text(xs[0] - 0.62, 1.68, "offline, once",
            fontsize=8, style="italic", color=SECONDARY)
    ax.text(xs[0] - 0.62, 0.43, "online, per batch",
            fontsize=8, style="italic", color=SECONDARY)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("wrote %s", out)


def geometry(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.5), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_xlim(-2.1, 3.1)
    ax.set_ylim(-2.2, 2.3)
    ax.axis("off")

    # level sets of the outputs: parallel lines, direction (cos70, sin70)
    direction = np.array([np.cos(np.radians(70)), np.sin(np.radians(70))])
    normal = np.array([-direction[1], direction[0]])
    for i in range(-2, 4):
        base = normal * (0.75 * i)
        p0, p1 = base - 2.6 * direction, base + 2.6 * direction
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#cccccc",
                linewidth=0.9, zorder=1)
    ax.text(2.28, 1.05, "level sets of\nthe outputs", fontsize=8,
            color=MUTED, rotation=70, ha="center", va="center")

    # reference activation cloud on one level set
    ax.add_patch(Ellipse((-0.55, 0.35), 1.7, 1.0, angle=25,
                         facecolor=BLUE, alpha=0.14, edgecolor="none",
                         zorder=2))
    ax.text(-0.55, 0.35, "reference", fontsize=8.5,
            color="#1c4f8f", ha="center", va="center", zorder=3)

    # visible drift: across the level sets
    tip = np.array([-0.15, 0.55]) + 1.55 * normal
    ax.add_patch(FancyArrowPatch((-0.15, 0.55), tuple(tip),
                                 arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.7, color=BLUE, zorder=4))
    ax.text(*(np.array([-0.15, 0.55]) + 0.85 * normal + [0.06, 0.16]),
            "visible drift", fontsize=8.5, color=BLUE, rotation=-18)

    # blind drift: along a level set, outputs frozen
    start = np.array([-0.75, -0.15])
    tip = start - 1.9 * direction
    ax.add_patch(FancyArrowPatch(tuple(start), tuple(tip),
                                 arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.7, color=RED, zorder=4))
    ax.text(*(start - 1.25 * direction + [0.14, 0.0]),
            "blind drift\n(outputs frozen)", fontsize=8.5, color=RED,
            ha="left", va="center")

    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("wrote %s", out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-dir", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    pipeline(args.figures_dir / "pipeline.png")
    geometry(args.figures_dir / "geometry_blind.png")


if __name__ == "__main__":
    main()
