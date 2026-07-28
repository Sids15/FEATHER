"""Symmetric-calibration robustness check for FEATHER (paper Sect. 6.7).

The deployed protocol calibrates FEATHER's reference mean and thresholds on a
held-out split of the *training* reference, while the output baselines are
calibrated on the clean deployment (test) stream. This script removes that
asymmetry: it recalibrates FEATHER and the PCA ablation on a held-out split of
the clean-test activations --- the exact protocol the output baselines use ---
and re-scores every episode, so the alarm-rate comparison is at matched
calibration.

It reads outputs/monitor_*/ runs that saved per-episode activations. The clean
episode is always present (`phi` saved by default); benign/gray/harmful drift
episodes need a run made with `--save-all-phi`. With clean `phi` only, the
script still reports the clean-stream false-alarm rate under symmetric
calibration; with all-phi runs it additionally reports the benign/gray/harmful
rates. NumPy only --- no GPU, no torch.

    python src/experiments/symmetric_feather.py                 # all runs
    python src/experiments/symmetric_feather.py --runs outputs/monitor_cifar10_seed0
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from feather.core.monitor import MonitorConfig, SubspaceDriftMonitor

logger = logging.getLogger("feather.experiments.symmetric")

CLEAN_EPISODE = {"cifar10c": "clean", "rotated_mnist": "rotation_00"}
BENIGN_MAX_DROP = 0.02
HARMFUL_MIN_DROP = 0.10


def orthonormal(basis: np.ndarray) -> np.ndarray:
    """Re-orthonormalize a float32-stored basis for the float64 monitor check."""
    q, _ = np.linalg.qr(basis.astype(np.float64))
    return q


def batch_alarm_rate(monitor: SubspaceDriftMonitor, phi: np.ndarray,
                     batch_size: int) -> float:
    """Fraction of size-``batch_size`` chunks of ``phi`` that alarm."""
    n_batches = max(len(phi) // batch_size, 1)
    alarms = sum(
        monitor.score(phi[i * batch_size:(i + 1) * batch_size]).alarm
        for i in range(n_batches)
    )
    return alarms / n_batches


def episode_accuracy(run_dir: Path) -> dict[str, float]:
    """Mean accuracy per episode from episodes.csv (labels for eval only)."""
    by_episode: dict[str, list[float]] = defaultdict(list)
    with (run_dir / "episodes.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_episode[row["episode"]].append(float(row["accuracy"]))
    return {ep: float(np.mean(v)) for ep, v in by_episode.items()}


def analyze_run(run_dir: Path) -> list[dict] | None:
    """Return per-episode symmetric alarm rates, or None if raw/ is absent."""
    raw_dir = run_dir / "raw"
    fit_path = run_dir / "fit.json"
    if not (raw_dir / "monitor.npz").exists() or not fit_path.exists():
        logger.warning("%s: no raw/monitor.npz — skipped", run_dir.name)
        return None
    fit = json.loads(fit_path.read_text())
    cfg = MonitorConfig(batch_size=fit["batch_size"], n_bootstrap=fit["n_bootstrap"],
                        quantile=fit["quantile"], seed=fit["seed"])
    mode = fit["mode"]
    clean_name = CLEAN_EPISODE[mode]
    clean_path = raw_dir / f"episode_{clean_name}.npz"
    clean = np.load(clean_path)
    if "phi" not in clean:
        logger.warning("%s: clean episode has no phi — rerun with save_phi", run_dir.name)
        return None

    arrays = np.load(raw_dir / "monitor.npz")
    blind = orthonormal(arrays["blind_basis"])
    pca = orthonormal(arrays["pca_basis"])
    clean_phi = clean["phi"].astype(np.float64)
    # random 50/50 split (not in-order: avoids any activation-ordering artifact),
    # matching split_reference_dataset — one half calibrates, the other evaluates
    perm = np.random.default_rng(fit["seed"]).permutation(len(clean_phi))
    half = len(clean_phi) // 2
    cal_phi, eval_phi = clean_phi[perm[:half]], clean_phi[perm[half:]]
    feather = SubspaceDriftMonitor(blind, cal_phi, cfg)
    pca_mon = SubspaceDriftMonitor(pca, cal_phi, cfg)

    accuracy = episode_accuracy(run_dir)
    clean_acc = accuracy[clean_name]
    seed = int("".join(c for c in run_dir.name.split("seed")[-1] if c.isdigit()))
    bs = fit["batch_size"]

    rows: list[dict] = []
    # clean false-alarm rate: evaluate on the held-out second half only
    rows.append({
        "mode": mode, "seed": seed, "episode": clean_name, "label": "clean",
        "feather_alarm_rate": batch_alarm_rate(feather, eval_phi, bs),
        "pca_alarm_rate": batch_alarm_rate(pca_mon, eval_phi, bs),
    })
    # drift episodes: only those whose phi was saved (--save-all-phi)
    for ep_path in sorted(raw_dir.glob("episode_*.npz")):
        episode = ep_path.stem[len("episode_"):]
        if episode == clean_name:
            continue
        data = np.load(ep_path)
        if "phi" not in data:
            continue  # benign/harmful phi only present after a --save-all-phi rerun
        drop = clean_acc - accuracy.get(episode, clean_acc)
        label = ("benign" if drop < BENIGN_MAX_DROP
                 else "harmful" if drop > HARMFUL_MIN_DROP else "gray")
        phi = data["phi"].astype(np.float64)
        rows.append({
            "mode": mode, "seed": seed, "episode": episode, "label": label,
            "feather_alarm_rate": batch_alarm_rate(feather, phi, bs),
            "pca_alarm_rate": batch_alarm_rate(pca_mon, phi, bs),
        })
    logger.info("%s: symmetric re-score over %d episodes with phi", run_dir.name, len(rows))
    return rows


def summarize(rows: list[dict]) -> dict:
    """mean ± std across seeds, per mode / category / monitor."""
    out: dict = {}
    for mode in sorted({r["mode"] for r in rows}):
        out[mode] = {}
        for monitor in ("feather", "pca"):
            out[mode][monitor] = {}
            for category in ("clean", "benign", "gray", "harmful"):
                per_seed: dict[int, list[float]] = defaultdict(list)
                for r in rows:
                    if r["mode"] == mode and r["label"] == category:
                        per_seed[r["seed"]].append(r[f"{monitor}_alarm_rate"])
                seed_means = [float(np.mean(v)) for v in per_seed.values()]
                if seed_means:
                    out[mode][monitor][category] = {
                        "mean": round(float(np.mean(seed_means)), 4),
                        "std": round(float(np.std(seed_means)), 4),
                        "n_seeds": len(seed_means),
                    }
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", type=Path, default=None,
                        help="run folders (default: every outputs/monitor_*)")
    parser.add_argument("--out", type=Path,
                        default=Path("paper/tables/symmetric_feather.json"))
    args = parser.parse_args()

    runs = args.runs or sorted(Path("outputs").glob("monitor_*"))
    rows: list[dict] = []
    for run in runs:
        result = analyze_run(run)
        if result:
            rows.extend(result)
    if not rows:
        logger.warning("no runs with saved activations found")
        return

    summary = summarize(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    for mode, monitors in summary.items():
        print(f"\n{mode} — symmetric (clean-test) calibration, mean±std over seeds:")
        print(f"  {'category':10s} {'FEATHER':>14s} {'PCA':>14s}")
        for category in ("clean", "benign", "gray", "harmful"):
            f = monitors["feather"].get(category)
            p = monitors["pca"].get(category)
            if f:
                print(f"  {category:10s} {f['mean']:.3f}±{f['std']:.3f}    "
                      f"{p['mean']:.3f}±{p['std']:.3f}  (n={f['n_seeds']})")
            else:
                print(f"  {category:10s} {'—  (needs --save-all-phi rerun)':>30s}")
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
