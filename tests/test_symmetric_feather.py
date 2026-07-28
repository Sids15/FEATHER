"""Tests for the symmetric-calibration FEATHER re-score (NumPy only)."""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "experiments"))

from symmetric_feather import analyze_run, batch_alarm_rate, orthonormal

from feather.core.monitor import MonitorConfig, SubspaceDriftMonitor

BATCH = 50
D = 8


def make_run(tmp_path: Path, save_drift_phi: bool = True) -> Path:
    """Synthetic run with clean-episode phi and one shifted drift episode."""
    run = tmp_path / "monitor_fake_seed2"
    raw = run / "raw"
    raw.mkdir(parents=True)
    rng = np.random.default_rng(0)

    basis = np.linalg.qr(rng.standard_normal((D, D)))[0]
    np.savez_compressed(
        raw / "monitor.npz",
        blind_basis=basis[:, :4].astype(np.float32),
        pca_basis=basis[:, 4:].astype(np.float32),
        weight=rng.normal(size=(10, D)).astype(np.float32),
        bias=np.zeros(10, dtype=np.float32),
    )
    clean_phi = rng.standard_normal((400, D)).astype(np.float16)
    np.savez_compressed(raw / "episode_clean.npz",
                        logits=rng.normal(size=(400, 10)).astype(np.float32),
                        labels=rng.integers(0, 10, 400).astype(np.int64),
                        phi=clean_phi)
    # a drift episode shifted far along the blind basis -> should alarm
    drift_phi = (rng.standard_normal((200, D)) + 8.0 * basis[:, 0]).astype(np.float16)
    kwargs = dict(logits=rng.normal(size=(200, 10)).astype(np.float32),
                  labels=rng.integers(0, 10, 200).astype(np.int64))
    if save_drift_phi:
        kwargs["phi"] = drift_phi
    np.savez_compressed(raw / "episode_shift_s5.npz", **kwargs)

    fieldnames = ["episode", "batch", "n", "accuracy"]
    rows = []
    for episode, acc in (("clean", 0.95), ("shift_s5", 0.60)):  # harmful drop
        for b in range(4):
            rows.append({"episode": episode, "batch": b, "n": BATCH, "accuracy": acc})
    with (run / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    (run / "fit.json").write_text(json.dumps({
        "mode": "cifar10c", "batch_size": BATCH,
        "n_bootstrap": 50, "quantile": 0.99, "seed": 2,
    }))
    return run


def test_orthonormal_restores_orthonormality():
    q = orthonormal(np.linalg.qr(np.random.default_rng(0).standard_normal((8, 5)))[0]
                    .astype(np.float32))
    assert np.allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-10)


def test_batch_alarm_rate_bounds():
    cfg = MonitorConfig(batch_size=BATCH, n_bootstrap=50, quantile=0.99, seed=0)
    basis = np.eye(D)[:, :4]
    ref = np.random.default_rng(1).standard_normal((400, D))
    monitor = SubspaceDriftMonitor(basis, ref, cfg)
    rate = batch_alarm_rate(monitor, ref, BATCH)
    assert 0.0 <= rate <= 1.0


class TestAnalyzeRun:
    def test_reports_clean_and_drift_with_phi(self, tmp_path):
        rows = analyze_run(make_run(tmp_path))
        by_ep = {r["episode"]: r for r in rows}
        assert set(by_ep) == {"clean", "shift_s5"}
        assert by_ep["clean"]["label"] == "clean"
        assert by_ep["shift_s5"]["label"] == "harmful"  # 35-pt drop
        # the far blind-subspace shift trips FEATHER; every rate is a fraction
        assert by_ep["shift_s5"]["feather_alarm_rate"] == 1.0
        for r in rows:
            assert 0.0 <= r["feather_alarm_rate"] <= 1.0
            assert 0.0 <= r["pca_alarm_rate"] <= 1.0

    def test_drift_without_phi_is_skipped(self, tmp_path):
        rows = analyze_run(make_run(tmp_path, save_drift_phi=False))
        assert {r["episode"] for r in rows} == {"clean"}  # only clean has phi

    def test_missing_monitor_npz_returns_none(self, tmp_path):
        run = tmp_path / "monitor_legacy_seed0"
        (run / "raw").mkdir(parents=True)
        (run / "fit.json").write_text("{}")
        assert analyze_run(run) is None
