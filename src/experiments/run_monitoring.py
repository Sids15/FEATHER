"""Run the FEATHER monitoring benchmark against a frozen trained model.

Produces the paper's raw per-batch records: true accuracy (eval-only labels),
output-based baselines (confidence, entropy), and the FEATHER + PCA-ablation
monitor statistics, for every episode of a drift benchmark.

Typical use on the workstation (after training, see docs/training.md):

    # Tier 2 — Rotated MNIST (angles 0..90 in steps of 15)
    python src/experiments/run_monitoring.py --model outputs/mnist_seed0/final_model.pt \
        --mode rotated_mnist --out-name monitor_mnist_seed0

    # Tier 3 — CIFAR-10-C (all 19 corruptions x 5 severities)
    python src/experiments/run_monitoring.py --model outputs/cifar10_seed0/final_model.pt \
        --mode cifar10c --out-name monitor_cifar10_seed0

By default the clean reference data is split 50/50 into a geometry split
(Fisher/PCA fit) and a held-out calibration split (reference mean, bootstrap
thresholds) — the corrected protocol of paper Sect. 6.7. Pass
--calibration-mode same_split only to reproduce the legacy (leaky) numbers.

Outputs: outputs/<out-name>/episodes.csv (one row per stream batch),
fit.json (subspace dimensions, thresholds, calibration mode, timing), plus a
full log in logs/<out-name>.log. Analysis/plots run later from episodes.csv
(no GPU needed), so only this script and training need the workstation.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import torch

from feather.data.vision import (
    CIFAR10C_CORRUPTIONS,
    SEVERITIES,
    cifar10_datasets,
    cifar10c_dataset,
    mnist_datasets,
    rotated_mnist_test,
)
from feather.monitoring import (
    fit_monitors,
    load_frozen_model,
    run_episode,
    split_reference_dataset,
)

logger = logging.getLogger("feather.experiments.monitoring")

ROTATION_ANGLES = (0, 15, 30, 45, 60, 75, 90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="path to a final_model.pt")
    parser.add_argument("--mode", required=True, choices=["rotated_mnist", "cifar10c"])
    parser.add_argument("--out-name", required=True, help="output folder name under outputs/")
    parser.add_argument("--batch-size", type=int, default=500, help="stream batch size")
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-mode", default="heldout",
                        choices=["heldout", "same_split"],
                        help="heldout (default): calibrate thresholds on a "
                        "reference split disjoint from the Fisher fit; "
                        "same_split: legacy leaky protocol, reproduction only")
    parser.add_argument("--calibration-fraction", type=float, default=0.5,
                        help="heldout only: share of reference data used for "
                        "threshold calibration")
    parser.add_argument("--split-seed", type=int, default=0,
                        help="heldout only: seed for the geometry/calibration split")
    parser.add_argument("--corruptions", nargs="*", default=None,
                        help="cifar10c only: subset of corruptions (default all 19)")
    parser.add_argument("--data-root", default=None, help="overrides FEATHER_DATA_DIR")
    parser.add_argument("--device", default=None, choices=[None, "cuda", "cpu"])
    return parser.parse_args()


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.getLogger().handlers[1].setLevel(logging.INFO)


def episodes_for(args: argparse.Namespace):
    """Yield (episode_name, dataset) pairs for the chosen benchmark."""
    if args.mode == "rotated_mnist":
        for angle in ROTATION_ANGLES:
            yield f"rotation_{angle:02d}", rotated_mnist_test(angle, args.data_root)
    else:
        corruptions = args.corruptions or list(CIFAR10C_CORRUPTIONS)
        unknown = set(corruptions) - set(CIFAR10C_CORRUPTIONS)
        if unknown:
            raise ValueError(f"unknown corruptions: {sorted(unknown)}")
        _, clean_test = cifar10_datasets(args.data_root, augment=False)
        yield "clean", clean_test
        for corruption in corruptions:
            for severity in SEVERITIES:
                yield (
                    f"{corruption}_s{severity}",
                    cifar10c_dataset(corruption, severity, args.data_root),
                )


def main() -> None:
    args = parse_args()
    out_dir = Path("outputs") / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(Path("logs") / f"{args.out_name}.log")
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("monitoring run '%s' | mode=%s | device=%s", args.out_name, args.mode, device)

    model = load_frozen_model(args.model, device)
    if args.mode == "rotated_mnist":
        reference, _ = mnist_datasets(args.data_root)
    else:
        reference, _ = cifar10_datasets(args.data_root, augment=False)

    if args.calibration_mode == "heldout":
        geometry, calibration = split_reference_dataset(
            reference, args.calibration_fraction, args.split_seed
        )
        logger.info(
            "held-out calibration: geometry n=%d, calibration n=%d (split seed %d)",
            len(geometry), len(calibration), args.split_seed,
        )
    else:
        geometry = calibration = reference
        logger.warning("legacy same_split calibration: thresholds will be optimistic")

    fit_start = time.perf_counter()
    bundle = fit_monitors(
        model, geometry, calibration, device,
        batch_size=args.batch_size, quantile=args.quantile,
        n_bootstrap=args.n_bootstrap, seed=args.seed,
    )
    fit_seconds = time.perf_counter() - fit_start
    logger.info("offline fit done in %.1fs (blind dim=%d)", fit_seconds, bundle.blind_dim)

    csv_path = out_dir / "episodes.csv"
    writer = None
    online_seconds = 0.0
    n_batches = 0
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        for episode, dataset in episodes_for(args):
            start = time.perf_counter()
            records = run_episode(
                model, bundle, dataset, device, episode, batch_size=args.batch_size
            )
            online_seconds += time.perf_counter() - start
            n_batches += len(records)
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
                writer.writeheader()
            writer.writerows(records)
            mean_acc = sum(r["accuracy"] for r in records) / len(records)
            alarms = sum(r["feather_alarm"] for r in records)
            logger.info(
                "episode %-24s | mean acc=%.4f | feather alarms=%d/%d",
                episode, mean_acc, alarms, len(records),
            )

    fit_info = {
        "model": str(args.model),
        "mode": args.mode,
        "batch_size": args.batch_size,
        "quantile": args.quantile,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "calibration_mode": args.calibration_mode,
        "calibration_fraction": args.calibration_fraction,
        "split_seed": args.split_seed,
        "geometry_n": bundle.geometry_n,
        "calibration_n": bundle.calibration_n,
        "blind_dim": bundle.blind_dim,
        "fisher_top_eigenvalues": [float(v) for v in bundle.fisher_eigenvalues[:12]],
        "feather_shift_threshold": bundle.feather.shift_threshold,
        "feather_energy_threshold": bundle.feather.energy_threshold,
        "pca_shift_threshold": bundle.pca.shift_threshold,
        "pca_energy_threshold": bundle.pca.energy_threshold,
        "output_baselines": {
            name: {
                "direction": b.direction,
                "threshold": b.threshold,
                "reference_mean": b.reference_mean,
                "params": b.params,
            }
            for name, b in bundle.baselines.items()
        },
        "offline_fit_seconds": round(fit_seconds, 2),
        "online_seconds_total": round(online_seconds, 2),
        "online_ms_per_batch": round(1000 * online_seconds / max(n_batches, 1), 2),
        "n_batches": n_batches,
        "device": device.type,
    }
    (out_dir / "fit.json").write_text(json.dumps(fit_info, indent=2))
    logger.info("done: %s (%d batches, %.1f ms/batch online)",
                csv_path, n_batches, fit_info["online_ms_per_batch"])


if __name__ == "__main__":
    main()
