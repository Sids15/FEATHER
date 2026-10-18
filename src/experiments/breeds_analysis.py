"""Score the BREEDS subpopulation-shift runs against the pre-declared criteria.

Criteria were fixed before any data was collected (docs/breeds-design.md):

- PRIMARY: some mixing fraction rho has an accuracy drop >= 10 points while
  every output baseline alarms on < 20% of batches and FEATHER on > 80% --
  harmful drift on real images with real labels that the outputs miss.
- SECONDARY: the smallest rho at which each monitor alarms on >= 50% of
  batches, and the accuracy drop already incurred there (earlier = better).
- DIAGNOSTIC: across the 20 swap episodes, the Spearman correlation of the
  accuracy drop with FEATHER's blind-shift score and blind fraction s, next
  to the output baselines' scores.

All monitors use the same 1% calibrated budget. The full-space mean shift is
recovered from FEATHER's logs exactly as in meanshift_baseline.py.

Output baselines must first be calibrated on held-out clean deployment data,
as for every other result in the paper (the training split is memorized):

    python src/experiments/refit_baselines.py --baseline-calibration clean-heldout --runs outputs/monitor_breeds_seed*
    python src/experiments/breeds_analysis.py outputs/monitor_breeds_seed*
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.meanshift_baseline import score_run

OUTPUT_MONITORS = ("atc", "conf", "entropy", "proxy")
ALL_MONITORS = ("feather", "meanshift", "pca") + OUTPUT_MONITORS
HARM_DROP, OUTPUT_QUIET, FEATHER_LOUD, DETECT = 0.10, 0.20, 0.80, 0.50


def spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def blind_fraction(monitor_dir: Path) -> dict[str, float]:
    s = defaultdict(list)
    with (monitor_dir / "episodes.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            s[row["episode"]].append(float(row["feather_direction_ratio"]))
    return {e: float(np.mean(v)) for e, v in s.items()}


def main() -> None:
    dirs = [Path(p) for p in sys.argv[1:]]
    if not dirs:
        sys.exit(__doc__)
    rows = []
    for d in dirs:
        if json.loads((d / "fit.json").read_text()).get("baseline_calibration") != "clean-heldout":
            sys.exit(f"{d}: run refit_baselines.py --baseline-calibration clean-heldout first")
        _, r, check = score_run(d)
        s = blind_fraction(d)
        for row in r:
            row["s"] = s[row["episode"]]
        rows += r
        print(f"{d.name}: mean-shift reconstruction rel. error {check['max_rel_err_vs_direct']:.1e}")

    by_ep = defaultdict(list)
    for row in rows:
        by_ep[row["episode"]].append(row)
    avg = lambda ep, key: float(np.mean([r[key] for r in by_ep[ep]]))

    sweep = sorted(e for e in by_ep if e.startswith("mix_"))
    table = [{"rho": int(e[4:]) / 100, "acc_drop": avg(e, "acc_drop"),
              **{m: avg(e, f"{m}_alarm_rate") for m in ALL_MONITORS}} for e in sweep]
    print(f"\n{'rho':>5} {'drop':>6}" + "".join(f"{m:>10}" for m in ALL_MONITORS))
    for t in table:
        print(f"{t['rho']:5.2f} {100 * t['acc_drop']:5.1f}p" + "".join(f"{t[m]:10.2f}" for m in ALL_MONITORS))

    primary = [t["rho"] for t in table if t["acc_drop"] >= HARM_DROP
               and t["feather"] > FEATHER_LOUD and all(t[m] < OUTPUT_QUIET for m in OUTPUT_MONITORS)]
    secondary = {}
    for m in ALL_MONITORS:
        hit = next((t for t in table if t[m] >= DETECT), None)
        secondary[m] = None if hit is None else {"rho": hit["rho"], "acc_drop_at_detection": round(hit["acc_drop"], 4)}

    swaps = [e for e in by_ep if e.startswith("swap_")]
    drop = [avg(e, "acc_drop") for e in swaps]
    diagnostic = {key: round(spearman(drop, [avg(e, key) for e in swaps]), 3)
                  for key in ("feather", "s", "confidence", "entropy", "atc", "proxy")}

    summary = {"n_seeds": len(dirs), "sweep": table,
               "primary_met": bool(primary), "primary_rhos": primary,
               "secondary_first_detection": secondary,
               "diagnostic_spearman_vs_acc_drop": diagnostic}
    print(f"\nPRIMARY met: {bool(primary)} {primary}")
    print("SECONDARY first rho with >=50% alarms:", json.dumps(secondary))
    print("DIAGNOSTIC Spearman(acc drop, score) over swap episodes:", json.dumps(diagnostic))
    out = Path("outputs") / "breeds_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print("written", out)


if __name__ == "__main__":
    main()
