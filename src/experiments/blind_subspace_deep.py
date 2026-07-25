"""Deep-feature silent-drift validation (peer-review major revision M1).

The paper's central claim — that drift confined to the Fisher blind subspace
is invisible to every output-based monitor yet caught by FEATHER — is proven
algebraically and confirmed on a 2-D linear testbed. This experiment confirms
it on a *trained deep network* (the CIFAR-10 ResNet-18), using the activations
and blind basis already saved under outputs/<run>/raw/ (no GPU, no rerun).

For a clean batch of penultimate activations phi, we inject a controlled shift
along a unit direction and measure, as a function of the shift magnitude:

- the largest change in any softmax output, mean confidence, and entropy
  (output-based monitors can only ever see these);
- FEATHER's and the covariance ablation's alarm behavior;
- for a semi-synthetic *harm* demonstration, the accuracy under a hidden label
  rule that reads the blind coordinate — degrading while the outputs stay frozen.

Two directions are compared at matched shift magnitude:
- blind:      a direction in the Fisher blind subspace (W u proportional to 1),
- sensitive:  a direction in its orthogonal complement (moves the outputs).

Run:  python src/experiments/blind_subspace_deep.py
Out:  outputs/blind_deep/blind_deep.csv, blind_deep_summary.json,
      paper/figures/blind_deep.png
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from feather.core.monitor import MonitorConfig, SubspaceDriftMonitor

logger = logging.getLogger("feather.experiments.blind_deep")

BLUE, VIOLET, RED, MUTED, INK = "#2a78d6", "#4a3aa7", "#e34948", "#898781", "#0b0b0b"
GRID, SURFACE, SECONDARY = "#e1e0d9", "#fcfcfb", "#52514e"


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def output_stats(phi: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> dict:
    prob = softmax(phi @ weight.T + bias)
    entropy = -(prob * np.log(prob + 1e-12)).sum(axis=1)
    return {"prob": prob, "conf": prob.max(axis=1), "entropy": entropy}


def sensitive_direction(blind_basis: np.ndarray, seed: int) -> np.ndarray:
    """A unit vector in the orthogonal complement of the blind subspace."""
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(blind_basis.shape[0])
    r -= blind_basis @ (blind_basis.T @ r)  # remove blind-subspace component
    return r / np.linalg.norm(r)


def run_seed(run_dir: Path, batch_size: int, alphas: np.ndarray,
             n_batches: int, seed: int) -> list[dict]:
    raw = run_dir / "raw"
    monitor = np.load(raw / "monitor.npz")
    calib = np.load(raw / "calibration.npz")
    weight, bias = monitor["weight"].astype(np.float64), monitor["bias"].astype(np.float64)
    # re-orthonormalize: bases were persisted as float32, so restore clean
    # double-precision orthonormality (same subspace) for the monitor's check
    blind_basis, _ = np.linalg.qr(monitor["blind_basis"].astype(np.float64))
    pca_basis, _ = np.linalg.qr(monitor["pca_basis"].astype(np.float64))
    phi_all = calib["phi"].astype(np.float64)

    config = MonitorConfig(batch_size=batch_size, n_bootstrap=500, quantile=0.99, seed=0)
    feather = SubspaceDriftMonitor(blind_basis, phi_all, config)
    pca = SubspaceDriftMonitor(pca_basis, phi_all, config)

    u_blind = blind_basis[:, 0] / np.linalg.norm(blind_basis[:, 0])
    u_sens = sensitive_direction(blind_basis, seed)
    # semi-synthetic harm (A-to-Z Option A): a hidden label rule that reads the
    # blind coordinate the model provably ignores. A sample's true label flips to
    # a definite wrong class once its (shifted) blind coordinate leaves the clean
    # tail (99th pct). Accuracy therefore collapses as blind drift grows, while
    # every model output stays exactly frozen.
    blind_coord = phi_all @ u_blind
    harm_threshold = float(np.percentile(blind_coord, 99))

    rng = np.random.default_rng(seed)
    rows = []
    for direction, u in (("blind", u_blind), ("sensitive", u_sens)):
        blind_component = float(u @ u_blind)  # how far the shift moves the blind coord
        for alpha in alphas:
            confs, ents, maxdp, f_alarms, p_alarms, hidden_acc = [], [], [], [], [], []
            for _ in range(n_batches):
                idx = rng.integers(0, len(phi_all), size=batch_size)
                phi = phi_all[idx]
                base = output_stats(phi, weight, bias)
                shifted = phi + alpha * u
                now = output_stats(shifted, weight, bias)
                maxdp.append(float(np.abs(now["prob"] - base["prob"]).max()))
                confs.append(float(now["conf"].mean() - base["conf"].mean()))
                ents.append(float(now["entropy"].mean() - base["entropy"].mean()))
                f_alarms.append(int(feather.score(shifted).alarm))
                p_alarms.append(int(pca.score(shifted).alarm))
                # hidden-label accuracy: predictions are frozen under blind drift,
                # but the true label flips once the shifted blind coordinate leaves
                # the clean tail, so accuracy falls while outputs do not move
                pred = now["prob"].argmax(axis=1)
                drifted = (blind_coord[idx] + alpha * blind_component) > harm_threshold
                hidden_label = np.where(drifted, (pred + 5) % 10, pred)
                hidden_acc.append(float((pred == hidden_label).mean()))
            rows.append({
                "seed": seed, "direction": direction, "alpha": round(float(alpha), 4),
                "max_output_prob_change": float(f"{np.mean(maxdp):.3e}"),
                "mean_conf_change": float(f"{np.mean(confs):.3e}"),
                "mean_entropy_change": float(f"{np.mean(ents):.3e}"),
                "feather_alarm_rate": round(float(np.mean(f_alarms)), 4),
                "pca_alarm_rate": round(float(np.mean(p_alarms)), 4),
                "hidden_label_accuracy": round(float(np.mean(hidden_acc)), 4),
            })
    return rows


def figure(rows: list[dict], out: Path) -> None:
    blind = [r for r in rows if r["direction"] == "blind"]
    by_alpha: dict[float, list[dict]] = {}
    for r in blind:
        by_alpha.setdefault(r["alpha"], []).append(r)
    alphas = sorted(by_alpha)

    def series(col):
        return np.array([np.mean([r[col] for r in by_alpha[a]]) for a in alphas])

    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)

    ax.plot(alphas, series("feather_alarm_rate"), color=VIOLET, linewidth=2.2,
            marker="o", markersize=4, label="FEATHER alarm rate")
    ax.plot(alphas, series("max_output_prob_change"), color=RED, linewidth=2.2,
            marker="s", markersize=4, label="max output-prob change")
    ax.plot(alphas, 1.0 - series("hidden_label_accuracy"), color=BLUE, linewidth=2.0,
            linestyle="--", marker="^", markersize=4, label="hidden-label error")
    ax.set_xlabel("blind-subspace shift magnitude  α", color=SECONDARY, fontsize=9)
    ax.set_ylabel("rate", color=SECONDARY, fontsize=9)
    ax.set_title("CIFAR-10 ResNet-18: blind-subspace drift is output-invisible,\n"
                 "FEATHER-visible, and (under a hidden rule) harmful",
                 color=INK, fontsize=10)
    ax.legend(loc="center right", fontsize=8, frameon=False, labelcolor=SECONDARY)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE); plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("."))
    parser.add_argument("--glob", default="outputs/monitor_cifar10_seed*")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--n-batches", type=int, default=20)
    parser.add_argument("--alphas", type=float, nargs="*",
                        default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3])
    args = parser.parse_args()

    run_dirs = sorted(p for p in (args.runs_root).glob(args.glob.replace("outputs/", "outputs/"))
                      if (p / "raw" / "monitor.npz").exists())
    if not run_dirs:
        raise SystemExit(f"no runs with raw/monitor.npz under {args.glob}")

    all_rows = []
    for run_dir in run_dirs:
        seed = int("".join(filter(str.isdigit, run_dir.name.split("seed")[-1])) or 0)
        rows = run_seed(run_dir, args.batch_size, np.array(args.alphas), args.n_batches, seed)
        all_rows.extend(rows)
        logger.info("blind-deep %s done (%d rows)", run_dir.name, len(rows))

    out_dir = args.runs_root / "outputs" / "blind_deep"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "blind_deep.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)

    # summary at the largest alpha, averaged over seeds, per direction
    top = max(args.alphas)
    summary = {"alpha_max": top, "n_seeds": len(run_dirs), "by_direction": {}}
    for direction in ("blind", "sensitive"):
        sel = [r for r in all_rows if r["direction"] == direction and r["alpha"] == top]
        summary["by_direction"][direction] = {
            "max_output_prob_change": float(f'{np.mean([r["max_output_prob_change"] for r in sel]):.3e}'),
            "mean_conf_change": float(f'{np.mean([r["mean_conf_change"] for r in sel]):.3e}'),
            "mean_entropy_change": float(f'{np.mean([r["mean_entropy_change"] for r in sel]):.3e}'),
            "feather_alarm_rate": round(float(np.mean([r["feather_alarm_rate"] for r in sel])), 4),
            "pca_alarm_rate": round(float(np.mean([r["pca_alarm_rate"] for r in sel])), 4),
            "hidden_label_accuracy": round(float(np.mean([r["hidden_label_accuracy"] for r in sel])), 4),
        }
    (out_dir / "blind_deep_summary.json").write_text(json.dumps(summary, indent=2))
    figure(all_rows, args.runs_root / "paper" / "figures" / "blind_deep.png")
    logger.info("summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
