"""Aggregate monitoring runs into the paper's tables and figures.

Reads every outputs/monitor_*/episodes.csv + fit.json under --runs-root,
labels episodes by *measured* accuracy drop relative to the clean episode
(benign < 2 points, harmful > 10 points, gray in between), scores each
detector with a threshold-free statistic (change vs. the clean episode), and
reports per-seed AUROC for harmful-episode detection.

Detectors compared:
- feather   : mean blind-subspace shift magnitude m_t (delta vs clean)
- pca       : same statistic from the covariance-ablation monitor
- confidence: drop in mean max-softmax confidence (output-based baseline)
- entropy   : rise in mean prediction entropy (output-based baseline)

Outputs: paper/tables/monitoring_summary.json,
paper/figures/results_cifar10c.png, paper/figures/results_rotated_mnist.png.
Run: python src/experiments/analyze_monitoring.py --runs-root <folder with outputs/>
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("feather.experiments.analysis")

BENIGN_MAX_DROP = 0.02
HARMFUL_MIN_DROP = 0.10
CLEAN_EPISODE = {"cifar10c": "clean", "rotated_mnist": "rotation_00"}

# Palette (dataviz reference, light mode).
BLUE, VIOLET, RED, MUTED, INK = "#2a78d6", "#4a3aa7", "#e34948", "#898781", "#0b0b0b"
GRID, BASELINE, SURFACE, SECONDARY = "#e1e0d9", "#c3c2b7", "#fcfcfb", "#52514e"
LABEL_COLOR = {"benign": BLUE, "gray": MUTED, "harmful": RED}


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney); labels are 1=harmful, 0=benign."""
    positives, negatives = scores[labels == 1], scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([positives, negatives]))) + 1
    rank_sum = ranks[: len(positives)].sum()
    u = rank_sum - len(positives) * (len(positives) + 1) / 2
    return float(u / (len(positives) * len(negatives)))


def load_episode_table(monitor_dir: Path) -> tuple[str, int, list[dict]]:
    """Return (mode, seed, per-episode aggregate rows) for one monitor run."""
    fit = json.loads((monitor_dir / "fit.json").read_text())
    # The model seed lives in the run-folder name (monitor_cifar10_seed3);
    # fit.json's "seed" field is the bootstrap seed, identical across runs.
    match = re.search(r"seed(\d+)", monitor_dir.name)
    fit["seed"] = int(match.group(1)) if match else fit["seed"]
    per_episode: dict[str, list[dict]] = defaultdict(list)
    with (monitor_dir / "episodes.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            per_episode[row["episode"]].append(row)

    clean_name = CLEAN_EPISODE[fit["mode"]]
    if clean_name not in per_episode:
        raise ValueError(f"{monitor_dir}: no clean episode {clean_name!r}")

    def mean(rows: list[dict], column: str) -> float:
        return float(np.mean([float(r[column]) for r in rows]))

    clean = per_episode[clean_name]
    baseline = {
        "acc": mean(clean, "accuracy"),
        "conf": mean(clean, "mean_confidence"),
        "ent": mean(clean, "mean_entropy"),
        "feather_m": mean(clean, "feather_shift_magnitude"),
        "pca_m": mean(clean, "pca_shift_magnitude"),
    }
    rows = []
    for episode, batch_rows in per_episode.items():
        accuracy = mean(batch_rows, "accuracy")
        drop = baseline["acc"] - accuracy
        label = (
            "benign" if drop < BENIGN_MAX_DROP
            else "harmful" if drop > HARMFUL_MIN_DROP
            else "gray"
        )
        rows.append({
            "mode": fit["mode"],
            "seed": fit["seed"],
            "episode": episode,
            "accuracy": accuracy,
            "acc_drop": drop,
            "label": label,
            "feather": mean(batch_rows, "feather_shift_magnitude") - baseline["feather_m"],
            "pca": mean(batch_rows, "pca_shift_magnitude") - baseline["pca_m"],
            "confidence": baseline["conf"] - mean(batch_rows, "mean_confidence"),
            "entropy": mean(batch_rows, "mean_entropy") - baseline["ent"],
            "feather_alarm_rate": mean(batch_rows, "feather_alarm"),
            "pca_alarm_rate": mean(batch_rows, "pca_alarm"),
        })
    return fit["mode"], fit["seed"], rows


def per_seed_alarm_rates(
    rows: list[dict], monitor: str, clean_name: str
) -> dict[str, dict]:
    """Batch alarm rates per seed and category, with mean ± std across seeds.

    Categories: the clean episode itself, plus benign / gray / harmful drift
    episodes (clean excluded from benign so the calibration check is visible).
    """
    by_seed: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_seed[row["seed"]].append(row)

    summary: dict[str, dict] = {}
    for category in ("clean", "benign", "gray", "harmful"):
        per_seed = {}
        for seed, seed_rows in sorted(by_seed.items()):
            if category == "clean":
                selected = [r for r in seed_rows if r["episode"] == clean_name]
            else:
                selected = [
                    r for r in seed_rows
                    if r["label"] == category and r["episode"] != clean_name
                ]
            if selected:
                rate = float(np.mean([r[f"{monitor}_alarm_rate"] for r in selected]))
                per_seed[str(seed)] = round(rate, 4)
        values = list(per_seed.values())
        summary[category] = {
            "mean": round(float(np.mean(values)), 4) if values else None,
            "std": round(float(np.std(values)), 4) if values else None,
            "per_seed": per_seed,
        }
    return summary


def paired_sign_flip_test(differences: list[float]) -> dict:
    """Exact two-sided sign-flip permutation test on per-seed differences.

    Enumerates all 2^n sign assignments (n = #seeds, so 32 for 5 seeds) and
    reports the fraction whose |mean| is at least the observed |mean|.
    """
    diffs = np.array(differences, dtype=float)
    n = len(diffs)
    observed = abs(diffs.mean())
    signs = np.array(
        [[1 if (mask >> i) & 1 else -1 for i in range(n)] for mask in range(2**n)]
    )
    flipped_means = np.abs((signs * diffs).mean(axis=1))
    return {
        "mean_difference": round(float(diffs.mean()), 4),
        "std_difference": round(float(diffs.std()), 4),
        "per_seed_differences": [round(float(d), 4) for d in diffs],
        "p_value": round(float(np.mean(flipped_means >= observed - 1e-12)), 4),
    }


def per_seed_aurocs(rows: list[dict], detector: str) -> list[float]:
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[row["seed"]].append(row)
    values = []
    for seed_rows in by_seed.values():
        scored = [r for r in seed_rows if r["label"] != "gray"]
        scores = np.array([r[detector] for r in scored])
        labels = np.array([1 if r["label"] == "harmful" else 0 for r in scored])
        values.append(auroc(scores, labels))
    return values


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def figure_cifar(rows: list[dict], out: Path) -> None:
    """Accuracy drop vs. FEATHER score, every CIFAR-10-C episode and seed."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axis(ax)
    for label in ("benign", "gray", "harmful"):
        subset = [r for r in rows if r["label"] == label]
        ax.scatter(
            [100 * r["acc_drop"] for r in subset],
            [r["feather"] for r in subset],
            s=14, color=LABEL_COLOR[label], alpha=0.65, linewidths=0,
            label=f"{label} ({len(subset)})",
        )
    ax.set_xlabel("measured accuracy drop (points)", color=SECONDARY, fontsize=9)
    ax.set_ylabel(r"FEATHER blind-shift score $\Delta m$", color=SECONDARY, fontsize=9)
    ax.set_title("CIFAR-10-C: blind-subspace shift tracks measured harm "
                 "(96 episodes x 5 seeds)", color=INK, fontsize=10)
    ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=SECONDARY)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def figure_mnist(rows: list[dict], out: Path) -> None:
    """Accuracy and detector scores as the rotation angle grows."""
    angles = sorted({int(r["episode"].split("_")[1]) for r in rows})

    def series(column: str) -> tuple[np.ndarray, np.ndarray]:
        means, stds = [], []
        for angle in angles:
            values = [r[column] for r in rows if int(r["episode"].split("_")[1]) == angle]
            means.append(np.mean(values))
            stds.append(np.std(values))
        return np.array(means), np.array(stds)

    fig, (left, right) = plt.subplots(1, 2, figsize=(8.4, 3.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    for ax in (left, right):
        style_axis(ax)
        ax.set_xlabel("rotation angle (degrees)", color=SECONDARY, fontsize=9)

    acc_mean, acc_std = series("accuracy")
    left.errorbar(angles, 100 * acc_mean, yerr=100 * acc_std, color=BLUE,
                  linewidth=2.0, capsize=3)
    left.set_ylabel("accuracy (%)", color=SECONDARY, fontsize=9)
    left.set_title("model accuracy under rotation", color=INK, fontsize=10)

    feather_mean, feather_std = series("feather")
    right.errorbar(angles, feather_mean, yerr=feather_std, color=VIOLET,
                   linewidth=2.0, capsize=3)
    right.set_ylabel(r"FEATHER score $\Delta m$", color=SECONDARY, fontsize=9)
    right.set_title("blind-subspace shift under rotation", color=INK, fontsize=10)

    fig.suptitle("Rotated MNIST (mean $\\pm$ std over 5 seeds)", color=INK, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("."),
                        help="folder containing outputs/monitor_*/")
    parser.add_argument("--figures-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument("--tables-dir", type=Path, default=Path("paper/tables"))
    args = parser.parse_args()

    all_rows: list[dict] = []
    calibration_modes: set[str] = set()
    for fit_path in sorted((args.runs_root / "outputs").glob("monitor_*/fit.json")):
        fit = json.loads(fit_path.read_text())
        # runs predating the held-out protocol carry no calibration_mode key
        calibration_modes.add(fit.get("calibration_mode", "same_split"))
        mode, seed, rows = load_episode_table(fit_path.parent)
        all_rows.extend(rows)
        logger.info("loaded %s (%s, seed %d): %d episodes", fit_path.parent.name,
                    mode, seed, len(rows))
    if calibration_modes - {"heldout"}:
        logger.warning(
            "some runs use same-split calibration %s — alarm-rate tables from "
            "these runs are legacy-only, not paper headline numbers",
            sorted(calibration_modes),
        )

    summary: dict = {
        "episode_counts": {}, "auroc": {}, "alarm_rates": {},
        "calibration_modes": sorted(calibration_modes),
    }
    for mode in ("cifar10c", "rotated_mnist"):
        rows = [r for r in all_rows if r["mode"] == mode]
        if not rows:
            continue
        counts = {label: sum(r["label"] == label for r in rows) for label in
                  ("benign", "gray", "harmful")}
        summary["episode_counts"][mode] = counts
        summary["auroc"][mode] = {}
        for detector in ("feather", "pca", "confidence", "entropy"):
            values = per_seed_aurocs(rows, detector)
            summary["auroc"][mode][detector] = {
                "mean": round(float(np.nanmean(values)), 4),
                "std": round(float(np.nanstd(values)), 4),
                "per_seed": [round(v, 4) for v in values],
            }
        clean_name = CLEAN_EPISODE[mode]
        feather_rates = per_seed_alarm_rates(rows, "feather", clean_name)
        pca_rates = per_seed_alarm_rates(rows, "pca", clean_name)
        entry = {"feather": feather_rates, "pca": pca_rates}
        benign_diffs = [
            pca_rates["benign"]["per_seed"][seed] - rate
            for seed, rate in feather_rates["benign"]["per_seed"].items()
            if seed in pca_rates["benign"]["per_seed"]
        ]
        if benign_diffs:
            entry["benign_pca_minus_feather"] = paired_sign_flip_test(benign_diffs)
        summary["alarm_rates"][mode] = entry

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    (args.tables_dir / "monitoring_summary.json").write_text(json.dumps(summary, indent=2))
    figure_cifar([r for r in all_rows if r["mode"] == "cifar10c"],
                 args.figures_dir / "results_cifar10c.png")
    figure_mnist([r for r in all_rows if r["mode"] == "rotated_mnist"],
                 args.figures_dir / "results_rotated_mnist.png")
    logger.info("summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
