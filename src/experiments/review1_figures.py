"""Generate the Review-I presentation figures from the tested Tier-1 prototype.

Outputs (docs/review1/figures/):
- geometry.png        — slide 6: the sensitive/blind split of activation space
- prototype_demo.png  — slide 11: blind drift is invisible to outputs but
                        caught by the calibrated FEATHER monitor

Run from the repo root:
    .venv/Scripts/python -m src.experiments.review1_figures
Deterministic (fixed seeds); regenerates both figures from scratch.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from feather.core.fisher import activation_fisher, fisher_subspaces
from feather.core.monitor import MonitorConfig, SubspaceDriftMonitor
from feather.data.synthetic2d import (
    DriftSpec,
    Synthetic2DConfig,
    Synthetic2DStream,
    bayes_boundary,
    benign_drift,
    harmful_drift,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "docs" / "review1" / "figures"
DPI = 200
ONSET = 15

# Validated palette (dataviz skill reference, light mode).
BLUE = "#2a78d6"  # class 0 / primary series
AQUA = "#1baf7a"  # class 1
VIOLET = "#4a3aa7"  # blind direction
RED = "#e34948"  # sensitive direction
CRITICAL = "#d03b3b"  # alarm status
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

CONFIG = Synthetic2DConfig(batch_size=500, n_batches=30, seed=42)


def style_axis(ax: plt.Axes) -> None:
    """Recessive chart chrome: hairline grid, muted ticks, no top/right spines."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fit_offline():
    """Reference data → head → Fisher subspaces → calibrated blind monitor."""
    w, b = bayes_boundary(CONFIG)
    weight = np.array([np.zeros(2), w])
    bias = np.array([0.0, b])
    reference = list(Synthetic2DStream(CONFIG, DriftSpec(kind="none")))
    phi = np.vstack([batch.x for batch in reference])
    y = np.concatenate([batch.y for batch in reference])
    sub = fisher_subspaces(activation_fisher(phi, y, weight, bias))
    monitor = SubspaceDriftMonitor(
        sub.blind_basis, phi, MonitorConfig(batch_size=500, n_bootstrap=300, seed=5)
    )
    return phi, y, weight, bias, sub, monitor


def mean_confidence(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> float:
    logits = x @ weight.T + bias
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    return float(p.max(axis=1).mean())


def figure_geometry(phi, y, sub) -> None:
    """Slide 6: two-class cloud, frozen boundary, sensitive vs blind directions."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    style_axis(ax)

    keep = np.random.default_rng(0).choice(len(phi), size=1500, replace=False)
    for cls, color in ((0, BLUE), (1, AQUA)):
        pts = phi[keep][y[keep] == cls]
        ax.scatter(pts[:, 0], pts[:, 1], s=6, color=color, alpha=0.35, linewidths=0)
    ax.text(-1.6, 1.35, "class 0", color=BLUE, fontsize=10, weight="bold")
    ax.text(1.1, 1.35, "class 1", color=AQUA, fontsize=10, weight="bold")

    ax.axvline(0.0, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(0.08, -1.78, "frozen decision boundary", color=SECONDARY, fontsize=8.5,
            rotation=90, va="bottom")

    arrows = (
        (sub.sensitive_basis[:, 0], RED, "sensitive direction\n(outputs react — existing\nmonitors can see drift here)"),
        (sub.blind_basis[:, 0], VIOLET, "blind direction — null(W)\n(outputs frozen: invisible to\nconfidence/entropy/ATC)"),
    )
    for direction, color, label in arrows:
        d = direction if direction[0] + direction[1] >= 0 else -direction
        ax.annotate(
            "", xy=1.6 * d, xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=color, linewidth=2.4,
                            mutation_scale=18),
        )
        offset = 1.85 * d
        ax.text(offset[0], offset[1], label, color=color, fontsize=8.5,
                ha="left" if d[0] > 0.5 else "center",
                va="center" if d[0] > 0.5 else "bottom")

    ax.set_xlim(-4.6, 4.6)
    ax.set_ylim(-2.0, 2.9)
    ax.set_aspect("equal")
    ax.set_xlabel("activation dimension 1", color=SECONDARY, fontsize=9)
    ax.set_ylabel("activation dimension 2", color=SECONDARY, fontsize=9)
    ax.set_title(
        "The Fisher matrix splits activation space into what the model can and cannot see",
        color=INK, fontsize=10.5, pad=12,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "geometry.png", facecolor=SURFACE)
    plt.close(fig)


def run_stream(drift: DriftSpec, weight, bias, monitor):
    """Per-batch confidence, monitor statistics, and alarms for one stream."""
    confidences, magnitudes, alarms = [], [], []
    for batch in Synthetic2DStream(CONFIG, drift):
        result = monitor.score(batch.x)
        confidences.append(mean_confidence(batch.x, weight, bias))
        magnitudes.append(result.shift_magnitude)
        alarms.append(result.shift_alarm)
    return np.array(confidences), np.array(magnitudes), np.array(alarms)


def figure_prototype_demo(weight, bias, monitor) -> None:
    """Slide 11: 2x2 small multiples — outputs vs FEATHER on both drift types."""
    streams = (
        ("Drift in blind subspace", benign_drift(CONFIG, magnitude=4.0, onset=ONSET)),
        ("Drift in sensitive direction", harmful_drift(CONFIG, magnitude=3.0, onset=ONSET)),
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 4.8), dpi=DPI, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    batches = np.arange(CONFIG.n_batches)

    results = [
        (title, *run_stream(drift, weight, bias, monitor)) for title, drift in streams
    ]
    # Shared bottom-row scale: unequal scales would make the near-zero right
    # panel look as active as the left one.
    magnitude_ceiling = 1.1 * max(m.max() for _, _, m, _ in results)

    for col, (title, confidence, magnitude, alarms) in enumerate(results):
        top, bottom = axes[0, col], axes[1, col]
        for ax in (top, bottom):
            style_axis(ax)
            ax.axvline(ONSET, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))

        top.plot(batches, confidence, color=BLUE, linewidth=2.0)
        top.set_ylim(0.5, 1.0)
        top.set_title(title, color=INK, fontsize=10)
        verdict_flat = abs(confidence[ONSET:].mean() - confidence[:ONSET].mean()) < 0.01
        top.text(0.03, 0.08,
                 "flat — output monitors see nothing" if verdict_flat else "moves — output monitors can react",
                 transform=top.transAxes, color=SECONDARY, fontsize=8)

        bottom.plot(batches, magnitude, color=VIOLET, linewidth=2.0)
        bottom.set_ylim(0.0, magnitude_ceiling)
        bottom.axhline(monitor.shift_threshold, color=MUTED, linewidth=1.0,
                       linestyle=(0, (2, 2)))
        if alarms.any():
            bottom.scatter(batches[alarms], magnitude[alarms], s=22, color=CRITICAL,
                           zorder=3, label="alarm")
            bottom.legend(loc="upper left", fontsize=8, frameon=False,
                          labelcolor=SECONDARY)
        else:
            bottom.text(0.03, 0.55,
                        "stays below threshold — drift barely\nprojects onto the blind subspace",
                        transform=bottom.transAxes, color=SECONDARY, fontsize=8)
        bottom.set_xlabel("stream batch", color=SECONDARY, fontsize=9)

    axes[0, 0].set_ylabel("mean softmax\nconfidence", color=SECONDARY, fontsize=9)
    axes[1, 0].set_ylabel("FEATHER blind-shift\nmagnitude $m_t$", color=SECONDARY, fontsize=9)
    axes[1, 0].text(0.03, 0.06, "calibrated threshold (99%)",
                    transform=axes[1, 0].transAxes, color=MUTED, fontsize=7.5)
    fig.suptitle(
        "Prototype output (38 passing tests): drift onset at batch 15 (dashed)",
        color=INK, fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "prototype_demo.png", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    phi, y, weight, bias, sub, monitor = fit_offline()
    figure_geometry(phi, y, sub)
    figure_prototype_demo(weight, bias, monitor)
    logger.info("figures written to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
