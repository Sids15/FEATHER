"""Tests for offline baseline refit from raw/ arrays (NumPy only)."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "experiments"))

from refit_baselines import refit_run

from feather.baselines import fit_output_baselines, score_output_baselines
from feather.core.monitor import MonitorConfig
from test_baselines import synthetic_probabilities

BATCH = 50
D = 8


def make_run(tmp_path: Path, with_baseline_columns=False) -> Path:
    """Synthetic monitor run: fit.json, episodes.csv, raw/ arrays."""
    run = tmp_path / "monitor_fake_seed0"
    raw = run / "raw"
    raw.mkdir(parents=True)
    rng = np.random.default_rng(0)

    probabilities, labels = synthetic_probabilities(n=1000, seed=1)
    np.savez_compressed(
        raw / "calibration.npz",
        logits=np.log(probabilities + 1e-12).astype(np.float32),
        labels=labels.astype(np.int64),
        phi=rng.normal(size=(1000, D)).astype(np.float16),
    )
    basis = np.eye(D, dtype=np.float32)
    np.savez_compressed(
        raw / "monitor.npz",
        blind_basis=basis[:, :4], pca_basis=basis[:, 4:],
        weight=rng.normal(size=(10, D)).astype(np.float32),
        bias=np.zeros(10, dtype=np.float32),
    )

    fieldnames = [
        "episode", "batch", "n", "accuracy",
        "feather_shift_magnitude", "feather_energy",
        "feather_shift_alarm", "feather_energy_alarm", "feather_alarm",
        "pca_shift_magnitude", "pca_energy",
        "pca_shift_alarm", "pca_energy_alarm", "pca_alarm",
    ]
    if with_baseline_columns:
        fieldnames += ["atc_score", "atc_alarm", "conf_score", "conf_alarm",
                       "entropy_score", "entropy_alarm", "proxy_score", "proxy_alarm"]
    rows = []
    for episode in ("clean", "drifted"):
        stream, stream_labels = synthetic_probabilities(
            n=2 * BATCH, sharpness=6.0 if episode == "clean" else 1.0,
            seed=hash(episode) % 1000,
        )
        np.savez_compressed(
            raw / f"episode_{episode}.npz",
            logits=np.log(stream + 1e-12).astype(np.float32),
            labels=stream_labels.astype(np.int64),
        )
        for b in range(2):
            row = {name: 0 for name in fieldnames}
            row.update(episode=episode, batch=b, n=BATCH, accuracy=0.9,
                       feather_shift_magnitude=0.5 + b, feather_energy=1.0,
                       pca_shift_magnitude=0.5 + b, pca_energy=1.0)
            rows.append(row)
    with (run / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    (run / "fit.json").write_text(json.dumps({
        "mode": "cifar10c",
        "batch_size": BATCH, "n_bootstrap": 50, "quantile": 0.99, "seed": 0,
        "feather_shift_threshold": 1.0, "feather_energy_threshold": 1.0,
        "pca_shift_threshold": 1.0, "pca_energy_threshold": 1.0,
    }))
    return run


class TestRefitRun:
    def test_appends_baseline_columns_to_legacy_csv(self, tmp_path):
        run = make_run(tmp_path, with_baseline_columns=False)
        assert refit_run(run)
        with (run / "episodes.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert "proxy_alarm" in rows[0] and "atc_score" in rows[0]
        assert all(r["conf_score"] != "" for r in rows)
        fit = json.loads((run / "fit.json").read_text())
        assert set(fit["output_baselines"]) == {"atc", "conf", "entropy", "proxy"}

    def test_clean_heldout_calibration_differs_and_records_mode(self, tmp_path):
        run = make_run(tmp_path)
        assert refit_run(run, baseline_calibration="clean-heldout")
        fit = json.loads((run / "fit.json").read_text())
        assert fit["baseline_calibration"] == "clean-heldout"
        clean_thr = fit["output_baselines"]["conf"]["threshold"]
        # train-split calibration yields a different threshold on the same run
        run2 = make_run(tmp_path / "b")
        refit_run(run2, baseline_calibration="train-split")
        train_thr = json.loads((run2 / "fit.json").read_text())["output_baselines"]["conf"]["threshold"]
        assert clean_thr != train_thr

    def test_matches_direct_computation(self, tmp_path):
        run = make_run(tmp_path)
        refit_run(run)
        calibration = np.load(run / "raw" / "calibration.npz")
        probabilities = np.exp(calibration["logits"])
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        config = MonitorConfig(batch_size=BATCH, n_bootstrap=50, quantile=0.99, seed=0)
        baselines = fit_output_baselines(probabilities, calibration["labels"], config)

        stream = np.load(run / "raw" / "episode_clean.npz")["logits"]
        first = np.exp(stream[:BATCH]); first /= first.sum(axis=1, keepdims=True)
        expected = score_output_baselines(baselines, first)
        with (run / "episodes.csv").open(encoding="utf-8") as f:
            row = next(csv.DictReader(f))
        for column, value in expected.items():
            assert float(row[column]) == pytest.approx(value, abs=1e-6)

    def test_quantile_override_rethresholds_monitors(self, tmp_path):
        run = make_run(tmp_path)
        refit_run(run, quantile=0.5)
        fit = json.loads((run / "fit.json").read_text())
        assert fit["quantile"] == 0.5
        assert fit["feather_shift_threshold"] != 1.0
        with (run / "episodes.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            stat = float(row["feather_shift_magnitude"])
            assert int(row["feather_shift_alarm"]) == int(
                stat > fit["feather_shift_threshold"]
            )
            assert int(row["feather_alarm"]) == int(
                int(row["feather_shift_alarm"]) or int(row["feather_energy_alarm"])
            )

    def test_missing_raw_is_skipped(self, tmp_path):
        run = tmp_path / "monitor_legacy_seed0"
        run.mkdir()
        (run / "fit.json").write_text("{}")
        assert not refit_run(run)

    def test_row_count_mismatch_raises(self, tmp_path):
        run = make_run(tmp_path)
        raw = np.load(run / "raw" / "episode_clean.npz")
        np.savez_compressed(run / "raw" / "episode_clean.npz",
                            logits=raw["logits"][:-7], labels=raw["labels"][:-7])
        with pytest.raises(ValueError, match="do not match"):
            refit_run(run)
