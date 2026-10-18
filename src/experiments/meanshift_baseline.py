"""Full-space mean-shift baseline, scored offline from existing monitoring runs.

The plain representation-space detector a reviewer asks for: alarm when the
batch activation mean drifts from the reference mean, ||mu_t - mu_ref|| (the
centroid displacement NEO re-centers by, and the cascade's stage-1 gate). It
needs no GPU rerun. FEATHER already logs, per batch, its blind magnitude
m = ||B^T dmu|| and direction ratio s = m / ||dmu||, so ||dmu|| = m / s exactly.
The threshold is bootstrapped on the saved calibration-split activations with
the same batch size, bootstrap count, quantile and seed FEATHER used, so the
comparison is at a matched false-alarm budget.

    python src/experiments/meanshift_baseline.py outputs/monitor_cifar10_seed*

Prints the clean/benign/gray/harmful alarm-rate table row (mean +- std over
seeds) next to FEATHER's, plus the paired FEATHER-vs-mean-shift benign test.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.analyze_monitoring import (
    CLEAN_EPISODE,
    load_episode_table,
    paired_sign_flip_test,
    per_seed_alarm_rates,
)
from feather.cascade import calibrate_shift_gate


def full_shift(row: dict) -> float:
    """||dmu|| recovered from FEATHER's logged m and s = m / ||dmu||."""
    s = float(row["feather_direction_ratio"])
    return float(row["feather_shift_magnitude"]) / s if s > 0 else 0.0


def score_run(monitor_dir: Path) -> tuple[str, list[dict], dict]:
    fit = json.loads((monitor_dir / "fit.json").read_text())
    calib = np.load(monitor_dir / "raw" / "calibration.npz")["phi"].astype(np.float64)
    mu_ref = calib.mean(axis=0)
    threshold = calibrate_shift_gate(
        calib, mu_ref, fit["batch_size"], fit["n_bootstrap"], fit["quantile"], fit["seed"]
    )

    per_episode = defaultdict(list)
    with (monitor_dir / "episodes.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            per_episode[row["episode"]].append(full_shift(row) > threshold)

    # exact check of the m/s reconstruction on the one episode whose phi was saved
    clean = CLEAN_EPISODE[fit["mode"]]
    phi = np.load(monitor_dir / "raw" / f"episode_{clean}.npz")["phi"].astype(np.float64)
    bs = fit["batch_size"]
    direct = [np.linalg.norm(phi[i:i + bs].mean(axis=0) - mu_ref)
              for i in range(0, len(phi), bs)]
    with (monitor_dir / "episodes.csv").open(encoding="utf-8") as f:
        logged = [full_shift(r) for r in csv.DictReader(f) if r["episode"] == clean]
    rel_err = float(np.max(np.abs(np.array(direct) - logged) / np.array(direct)))

    mode, _, rows = load_episode_table(monitor_dir)
    for row in rows:
        row["meanshift_alarm_rate"] = float(np.mean(per_episode[row["episode"]]))
    return mode, rows, {"threshold": threshold, "max_rel_err_vs_direct": rel_err}


def main() -> None:
    dirs = [Path(p) for p in sys.argv[1:]]
    if not dirs:
        sys.exit(__doc__)
    all_rows, mode = [], None
    for d in dirs:
        mode, rows, check = score_run(d)
        all_rows += rows
        print(f"{d.name}: threshold={check['threshold']:.4f}  "
              f"max rel. error of m/s vs direct ||dmu|| = {check['max_rel_err_vs_direct']:.2e}")

    clean = CLEAN_EPISODE[mode]
    table = {m: per_seed_alarm_rates(all_rows, m, clean)
             for m in ("meanshift", "feather", "pca")}
    print(f"\n{'monitor':12s}" + "".join(f"{c:>16s}" for c in ("clean", "benign", "gray", "harmful")))
    for name, summary in table.items():
        cells = "".join(f"{summary[c]['mean']:>9.2f} ± {summary[c]['std']:.2f}" for c in summary)
        print(f"{name:12s}{cells}")

    seeds = sorted(table["feather"]["benign"]["per_seed"])
    diffs = [table["meanshift"]["benign"]["per_seed"][s] - table["feather"]["benign"]["per_seed"][s]
             for s in seeds]
    print("\nbenign: mean-shift minus FEATHER, paired over seeds:",
          json.dumps(paired_sign_flip_test(diffs)))


if __name__ == "__main__":
    main()
