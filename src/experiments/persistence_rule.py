"""k-consecutive persistence rule, scored offline from existing monitoring runs.

A monitor raises an alarm at batch t only if its per-batch alarm fired on
batches t-k+1..t of the same episode stream -- the standard run rule for
suppressing isolated false alarms in control charts. Applied identically to
every monitor (FEATHER, covariance ablation, full-space mean shift, and the four
output baselines), so the comparison stays at a matched rule. Also reports the
mean detection delay on harmful episodes (batches until the first alarm).

    python src/experiments/persistence_rule.py outputs/monitor_cifar10_seed*
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.analyze_monitoring import CLEAN_EPISODE, load_episode_table, per_seed_alarm_rates
from experiments.meanshift_baseline import full_shift
from feather.cascade import calibrate_shift_gate

MONITORS = ("feather", "pca", "meanshift", "atc", "conf", "entropy", "proxy")
KS = (1, 2, 3, 4)


def persistent(alarms: list[int], k: int) -> list[int]:
    """1 at t iff the last k raw alarms (t-k+1..t) all fired."""
    run, out = 0, []
    for a in alarms:
        run = run + 1 if a else 0
        out.append(int(run >= k))
    return out


def score_run(monitor_dir: Path) -> tuple[str, list[dict]]:
    fit = json.loads((monitor_dir / "fit.json").read_text())
    calib = np.load(monitor_dir / "raw" / "calibration.npz")["phi"].astype(np.float64)
    ms_thr = calibrate_shift_gate(calib, calib.mean(axis=0), fit["batch_size"],
                                  fit["n_bootstrap"], fit["quantile"], fit["seed"])
    raw = defaultdict(lambda: defaultdict(list))  # episode -> monitor -> per-batch alarms
    with (monitor_dir / "episodes.csv").open(encoding="utf-8") as f:
        for row in sorted(csv.DictReader(f), key=lambda r: (r["episode"], int(r["batch"]))):
            for m in MONITORS:
                a = int(full_shift(row) > ms_thr) if m == "meanshift" else int(row[f"{m}_alarm"])
                raw[row["episode"]][m].append(a)

    mode, _, rows = load_episode_table(monitor_dir)
    for row in rows:
        for m in MONITORS:
            for k in KS:
                fired = persistent(raw[row["episode"]][m], k)
                row[f"{m}_k{k}_alarm_rate"] = float(np.mean(fired))
                row[f"{m}_k{k}_delay"] = fired.index(1) if 1 in fired else len(fired)
    return mode, rows


def main() -> None:
    dirs = [Path(p) for p in sys.argv[1:]]
    if not dirs:
        sys.exit(__doc__)
    rows, mode = [], None
    for d in dirs:
        mode, r = score_run(d)
        rows += r
    clean = CLEAN_EPISODE[mode]
    harmful = [r for r in rows if r["label"] == "harmful" and r["episode"] != clean]
    print(f"{'monitor':10s} k" + "".join(f"{c:>15s}" for c in ("clean", "benign", "gray", "harmful"))
          + "   harmful delay (batches)")
    for m in MONITORS:
        for k in KS:
            s = per_seed_alarm_rates(rows, f"{m}_k{k}", clean)
            cells = "".join(f"{s[c]['mean']:>8.2f} ± {s[c]['std']:.2f}" for c in s)
            delay = np.mean([r[f"{m}_k{k}_delay"] for r in harmful])
            print(f"{m:10s} {k}{cells}   {delay:6.2f}")
        print()


if __name__ == "__main__":
    main()
