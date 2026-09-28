"""FEATHER-Lite cascade self-check on a synthetic stream with a known blind
subspace. No River, no GPU: the stage-1 gate is the self-contained full-space
shift threshold, stage 2 is the real SubspaceDriftMonitor.

The three behaviours the cascade must have:
  * clean stream          -> no cascade alarm, FEATHER rarely runs;
  * blind-subspace drift  -> gate fires, FEATHER confirms, cascade alarms;
  * sensitive drift        -> gate fires, but FEATHER defers, no cascade alarm.
"""

from __future__ import annotations

import numpy as np

from feather.cascade import calibrate_shift_gate, run_cascade, summarize
from feather.core.monitor import MonitorConfig, SubspaceDriftMonitor

D = 6
BLIND_DIM = 3
BATCH = 200
N_REF = 4000


def _fixture():
    rng = np.random.default_rng(0)
    blind_basis = np.eye(D)[:, :BLIND_DIM]  # watch e0,e1,e2 (orthonormal)
    reference = rng.standard_normal((N_REF, D))
    config = MonitorConfig(batch_size=BATCH, n_bootstrap=300, quantile=0.99, seed=0)
    feather = SubspaceDriftMonitor(blind_basis, reference, config)
    mu_ref = reference.mean(axis=0)
    tau = calibrate_shift_gate(reference, mu_ref, BATCH, n_bootstrap=300, seed=0)
    return rng, feather, mu_ref, tau


def _stream(rng, mean_shift, k):
    return [rng.standard_normal((BATCH, D)) + mean_shift for _ in range(k)]


def test_clean_stream_stays_quiet():
    rng, feather, mu_ref, tau = _fixture()
    records = run_cascade(feather, mu_ref, tau, _stream(rng, np.zeros(D), 8))
    summary = summarize(records, onset=None)
    assert summary["cascade_alarm_rate"] == 0.0
    assert summary["feather_run_rate"] <= 0.2  # gate almost never fires on clean


def test_blind_drift_is_caught():
    rng, feather, mu_ref, tau = _fixture()
    shift = np.zeros(D)
    shift[0] = 1.5  # along a watched (blind) direction
    stream = _stream(rng, np.zeros(D), 4) + _stream(rng, shift, 4)
    records = run_cascade(feather, mu_ref, tau, stream)
    summary = summarize(records, onset=4)
    assert summary["detection_delay"] == 0  # gate fires the moment drift starts
    assert any(r.cascade_alarm for r in records)  # blind -> alarm
    assert summary["feather_run_rate"] < 1.0  # cheaper than FEATHER-always


def test_sensitive_drift_is_deferred():
    rng, feather, mu_ref, tau = _fixture()
    shift = np.zeros(D)
    shift[3] = 1.5  # orthogonal to the blind subspace (output-visible)
    stream = _stream(rng, np.zeros(D), 4) + _stream(rng, shift, 4)
    records = run_cascade(feather, mu_ref, tau, stream)
    summary = summarize(records, onset=4)
    assert summary["detection_delay"] == 0  # gate still fires on the full-space shift
    assert summary["cascade_alarm_rate"] == 0.0  # but FEATHER defers, no cascade alarm
