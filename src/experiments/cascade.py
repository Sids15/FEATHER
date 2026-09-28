"""Run the FEATHER-Lite cascade benchmark against a frozen trained model.

The cascade (src/feather/cascade.py) runs a cheap full-space shift gate on every
batch and only calls the FEATHER blind-subspace monitor when the gate fires. For
each drift episode we build a clean->drift stream (a short clean prefix, then the
episode) so detection delay is measurable, then report, per episode: the fraction
of batches FEATHER actually ran (the cost win), the cascade's alarm rate, and its
detection delay. FEATHER-always is scored on the same batches for comparison.

Typical use on the workstation (after training + one monitoring run exist):

    python src/experiments/cascade.py --model outputs/cifar10_seed0/final_model.pt \
        --mode cifar10c --out-name cascade_cifar10_seed0

    python src/experiments/cascade.py --model outputs/mnist_seed0/final_model.pt \
        --mode rotated_mnist --out-name cascade_mnist_seed0

Outputs: outputs/<out-name>/episodes.csv (one row per stream batch) and
summary.json (per-grade aggregates). Grading (benign/gray/harmful) uses the same
measured accuracy-drop thresholds as the main monitoring benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from feather.cascade import CascadeMonitor, calibrate_shift_gate, summarize
from feather.data.vision import (
    CIFAR10C_CORRUPTIONS,
    SEVERITIES,
    cifar10_datasets,
    cifar10c_dataset,
    mnist_datasets,
    rotated_mnist_test,
)
from feather.monitoring import (
    extract_activations,
    fit_monitors,
    load_frozen_model,
    split_reference_dataset,
)

logger = logging.getLogger("feather.experiments.cascade")

ROTATION_ANGLES = (0, 15, 30, 45, 60, 75, 90)
BENIGN_MAX_DROP = 2.0  # accuracy-drop points; < this is benign
HARMFUL_MIN_DROP = 10.0  # > this is harmful; between is gray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="path to a final_model.pt")
    parser.add_argument("--mode", required=True, choices=["rotated_mnist", "cifar10c"])
    parser.add_argument("--out-name", required=True)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--clean-prefix", type=int, default=5,
                        help="clean batches prepended to each episode (onset marker)")
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--corruptions", nargs="*", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default=None, choices=[None, "cuda", "cpu"])
    return parser.parse_args()


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )


def episodes_for(args, data_root):
    """Yield (episode_name, dataset). First item is the clean episode."""
    if args.mode == "rotated_mnist":
        for angle in ROTATION_ANGLES:
            yield f"rotation_{angle:02d}", rotated_mnist_test(angle, data_root)
    else:
        corruptions = args.corruptions or list(CIFAR10C_CORRUPTIONS)
        _, clean_test = cifar10_datasets(data_root, augment=False)
        yield "clean", clean_test
        for corruption in corruptions:
            for severity in SEVERITIES:
                yield (f"{corruption}_s{severity}",
                       cifar10c_dataset(corruption, severity, data_root))


def chunk(array: np.ndarray, size: int) -> list[np.ndarray]:
    """Split rows into full batches of `size` (drop a short remainder)."""
    n = (array.shape[0] // size) * size
    return [array[i:i + size] for i in range(0, n, size)]


def grade(drop: float) -> str:
    if drop < BENIGN_MAX_DROP:
        return "benign"
    if drop > HARMFUL_MIN_DROP:
        return "harmful"
    return "gray"


def main() -> None:
    args = parse_args()
    out_dir = Path("outputs") / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(Path("logs") / f"{args.out_name}.log")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("cascade run '%s' | mode=%s | device=%s", args.out_name, args.mode, device)

    model = load_frozen_model(args.model, device)
    if args.mode == "rotated_mnist":
        reference, clean_test = mnist_datasets(args.data_root)
    else:
        reference, clean_test = cifar10_datasets(args.data_root, augment=False)

    geometry, calibration = split_reference_dataset(
        reference, args.calibration_fraction, args.split_seed)
    bundle = fit_monitors(model, geometry, calibration, device,
                          batch_size=args.batch_size, quantile=args.quantile,
                          n_bootstrap=args.n_bootstrap, seed=args.seed)
    mu_ref = bundle.calibration_phi.mean(axis=0)
    tau = calibrate_shift_gate(bundle.calibration_phi, mu_ref, args.batch_size,
                               n_bootstrap=args.n_bootstrap, quantile=args.quantile,
                               seed=args.seed)
    logger.info("blind dim=%d | stage-1 gate threshold=%.6g", bundle.blind_dim, tau)

    # a clean prefix drawn from clean deployment data (disjoint from the fit)
    clean_phi, clean_logits, clean_labels = extract_activations(model, clean_test, device)
    clean_acc = float((clean_logits.argmax(1) == clean_labels).mean())
    prefix = chunk(clean_phi, args.batch_size)[: args.clean_prefix]
    if len(prefix) < args.clean_prefix:
        raise ValueError("not enough clean data for the requested --clean-prefix")

    csv_path = out_dir / "episodes.csv"
    writer = None
    per_grade: dict[str, dict[str, list]] = {}
    online_seconds = 0.0
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        for episode, dataset in episodes_for(args, args.data_root):
            phi, logits, labels = extract_activations(model, dataset, device)
            acc = float((logits.argmax(1) == labels).mean())
            drop = 100.0 * (clean_acc - acc)
            episode_grade = "clean" if episode in ("clean", "rotation_00") else grade(drop)
            drift_batches = chunk(phi, args.batch_size)
            stream = prefix + drift_batches
            onset = len(prefix)

            monitor = CascadeMonitor(bundle.feather, mu_ref, tau)
            start = time.perf_counter()
            records = [monitor.process_batch(b, batch=i) for i, b in enumerate(stream)]
            online_seconds += time.perf_counter() - start
            # FEATHER-always, for the cost/coverage comparison
            always = [int(bundle.feather.score(b).alarm) for b in stream]

            summary = summarize(records, onset=onset)
            rows = []
            for r, a in zip(records, always):
                rows.append({
                    "episode": episode, "grade": episode_grade, "batch": r.batch,
                    "onset": onset, "accuracy_drop": round(drop, 3),
                    "stage1_shift": r.stage1_shift, "stage1_flag": int(r.stage1_flag),
                    "feather_ran": int(r.feather_ran), "feather_alarm": int(r.feather_alarm),
                    "cascade_alarm": int(r.cascade_alarm), "feather_always_alarm": a,
                })
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
            writer.writerows(rows)

            bucket = per_grade.setdefault(episode_grade, {"run": [], "delay": [], "cascade": [], "always": []})
            bucket["run"].append(summary["feather_run_rate"])
            bucket["cascade"].append(summary["cascade_alarm_rate"])
            bucket["always"].append(float(np.mean(always)))
            if summary["detection_delay"] is not None:
                bucket["delay"].append(summary["detection_delay"])
            logger.info("episode %-22s [%-7s] drop=%.1f | feather ran %.0f%% | delay=%s",
                        episode, episode_grade, drop,
                        100 * summary["feather_run_rate"], summary["detection_delay"])

    def agg(values):
        return round(float(np.mean(values)), 4) if values else None

    out = {"suite": args.mode, "model": str(args.model), "blind_dim": bundle.blind_dim,
           "stage1_threshold": tau, "clean_prefix": args.clean_prefix, "by_grade": {}}
    for g, b in per_grade.items():
        out["by_grade"][g] = {
            "n_episodes": len(b["run"]),
            "feather_run_rate": agg(b["run"]),          # cost: fraction FEATHER ran
            "cascade_alarm_rate": agg(b["cascade"]),
            "feather_always_alarm_rate": agg(b["always"]),  # coverage comparison
            "detection_delay": agg(b["delay"]),         # batches from onset
        }
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2))
    logger.info("done: %s | %s", csv_path, json.dumps(out["by_grade"]))


if __name__ == "__main__":
    main()
