"""Refit output baselines (and optionally thresholds) without a GPU rerun.

Monitoring runs made with --save-raw (the default) persist per-episode
logits/labels and the calibration arrays under outputs/<run>/raw/. This
script refits every output baseline from those arrays and rewrites the
baseline columns of episodes.csv in place — so changing a baseline, adding a
new one, or recalibrating requires only this NumPy-only script, never
another pass over the datasets on the workstation.

    python src/experiments/refit_baselines.py                 # all runs
    python src/experiments/refit_baselines.py --runs outputs/monitor_cifar10_seed0
    python src/experiments/refit_baselines.py --quantile 0.995 # also re-thresholds
                                                               # the subspace monitors

With --quantile, the FEATHER/PCA bootstrap thresholds are recomputed from
the saved calibration activations and every monitor alarm column is
re-derived from the (unchanged) statistic columns; fit.json is updated to
match. Runs without a raw/ directory (legacy runs) are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from feather.baselines import fit_output_baselines, score_output_baselines
from feather.core.monitor import MonitorConfig, SubspaceDriftMonitor

logger = logging.getLogger("feather.experiments.refit")

_BASELINE_COLUMNS = [
    f"{name}_{kind}"
    for name in ("atc", "conf", "entropy", "proxy")
    for kind in ("score", "alarm")
]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _batch_slices(rows: list[dict]) -> list[slice]:
    """Row-order slices into an episode's raw arrays, from the CSV 'n' column."""
    slices, start = [], 0
    for row in rows:
        n = int(row["n"])
        slices.append(slice(start, start + n))
        start += n
    return slices


def refit_run(run_dir: Path, quantile: float | None = None) -> bool:
    """Refit one run from its raw/ arrays; returns False when raw/ is absent."""
    raw_dir = run_dir / "raw"
    fit_path = run_dir / "fit.json"
    if not (raw_dir / "calibration.npz").exists() or not fit_path.exists():
        logger.warning("%s: no raw/ arrays (legacy run?) — skipped", run_dir.name)
        return False

    fit = json.loads(fit_path.read_text())
    config = MonitorConfig(
        batch_size=fit["batch_size"],
        n_bootstrap=fit["n_bootstrap"],
        quantile=quantile if quantile is not None else fit["quantile"],
        seed=fit["seed"],
    )
    calibration = np.load(raw_dir / "calibration.npz")
    baselines = fit_output_baselines(
        _softmax(calibration["logits"]), calibration["labels"], config
    )

    monitors = {}
    if quantile is not None:
        arrays = np.load(raw_dir / "monitor.npz")
        phi = calibration["phi"].astype(np.float64)
        monitors = {
            "feather": SubspaceDriftMonitor(arrays["blind_basis"].astype(np.float64), phi, config),
            "pca": SubspaceDriftMonitor(arrays["pca_basis"].astype(np.float64), phi, config),
        }

    csv_path = run_dir / "episodes.csv"
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    fieldnames += [c for c in _BASELINE_COLUMNS if c not in fieldnames]

    by_episode: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_episode[row["episode"]].append(row)

    for episode, episode_rows in by_episode.items():
        raw = np.load(raw_dir / f"episode_{episode}.npz")
        logits = raw["logits"]
        if logits.shape[0] != sum(int(r["n"]) for r in episode_rows):
            raise ValueError(
                f"{run_dir.name}/{episode}: raw rows ({logits.shape[0]}) do not "
                "match the episode's CSV batch sizes"
            )
        for row, batch in zip(episode_rows, _batch_slices(episode_rows)):
            row.update(score_output_baselines(baselines, _softmax(logits[batch])))
            for prefix, monitor in monitors.items():
                shift_alarm = float(row[f"{prefix}_shift_magnitude"]) > monitor.shift_threshold
                energy_alarm = float(row[f"{prefix}_energy"]) > monitor.energy_threshold
                row[f"{prefix}_shift_alarm"] = int(shift_alarm)
                row[f"{prefix}_energy_alarm"] = int(energy_alarm)
                row[f"{prefix}_alarm"] = int(shift_alarm or energy_alarm)

    tmp_path = csv_path.with_suffix(".csv.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, csv_path)

    fit["output_baselines"] = {
        name: {
            "direction": b.direction,
            "threshold": b.threshold,
            "reference_mean": b.reference_mean,
            "params": b.params,
        }
        for name, b in baselines.items()
    }
    if quantile is not None:
        fit["quantile"] = quantile
        for prefix, monitor in monitors.items():
            fit[f"{prefix}_shift_threshold"] = monitor.shift_threshold
            fit[f"{prefix}_energy_threshold"] = monitor.energy_threshold
    fit_path.write_text(json.dumps(fit, indent=2))
    logger.info(
        "%s: refit %d baselines over %d batches%s",
        run_dir.name, len(baselines), len(rows),
        "" if quantile is None else f", monitors re-thresholded at q={quantile}",
    )
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", type=Path, default=None,
                        help="run folders (default: every outputs/monitor_*)")
    parser.add_argument("--quantile", type=float, default=None,
                        help="also recalibrate the FEATHER/PCA thresholds at "
                        "this quantile and re-derive their alarm columns")
    args = parser.parse_args()

    runs = args.runs or sorted(Path("outputs").glob("monitor_*"))
    refitted = sum(refit_run(run, args.quantile) for run in runs)
    logger.info("refit %d/%d runs", refitted, len(runs))


if __name__ == "__main__":
    main()
