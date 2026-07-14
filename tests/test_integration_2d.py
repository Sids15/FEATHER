"""End-to-end Tier-1 validation: 2D stream → activation Fisher → drift monitor.

In the linear 2D toy the geometry is known exactly: the Fisher sensitive
direction is the boundary normal w, the blind direction is boundary-parallel.
So drift built as `benign_drift` (boundary-parallel) moves *entirely inside the
blind subspace* — provably invisible to softmax outputs (Prop. 1) yet caught by
the blind-subspace monitor — while `harmful_drift` (along w) is output-visible
and lands in the sensitive subspace.

Note (docs/revised-plan.md §1 P2): in this linear toy, blind-subspace drift is
*benign* because φ = x has no feature extractor to corrupt. The toy validates
the geometry and statistics; the blind-drift ↔ harm correlation for deep nets
is exactly what Tiers 2-3 test empirically.
"""

import numpy as np
import pytest

from feather.core.fisher import activation_fisher, fisher_subspaces
from feather.core.monitor import MonitorConfig, SubspaceDriftMonitor
from feather.data.synthetic2d import (
    DriftSpec,
    Synthetic2DConfig,
    Synthetic2DStream,
    bayes_boundary,
    benign_drift,
    harmful_drift,
)


@pytest.fixture(scope="module")
def setup():
    cfg = Synthetic2DConfig(batch_size=500, n_batches=30, seed=42)
    w, b = bayes_boundary(cfg)
    weight = np.array([np.zeros(2), w])  # binary softmax head equivalent
    bias = np.array([0.0, b])

    reference_batches = list(Synthetic2DStream(cfg, DriftSpec(kind="none")))
    phi = np.vstack([batch.x for batch in reference_batches])
    y = np.concatenate([batch.y for batch in reference_batches])

    fisher = activation_fisher(phi, y, weight, bias)
    sub = fisher_subspaces(fisher)
    monitor = SubspaceDriftMonitor(
        sub.blind_basis, phi, MonitorConfig(batch_size=500, n_bootstrap=300, seed=5)
    )
    return cfg, w, weight, bias, sub, monitor


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def test_blind_subspace_is_boundary_parallel(setup):
    cfg, w, _, _, sub, _ = setup
    w_parallel = np.array([-w[1], w[0]]) / np.linalg.norm(w)
    cosine = abs(sub.blind_basis[:, 0] @ w_parallel)
    assert cosine == pytest.approx(1.0, abs=1e-6)


def test_blind_drift_is_invisible_to_outputs_but_caught_by_monitor(setup):
    cfg, _, weight, bias, _, monitor = setup
    drift = benign_drift(cfg, magnitude=4.0, onset=15)
    post_onset = [b for b in Synthetic2DStream(cfg, drift) if b.index >= 15]

    for batch in post_onset:
        # Proposition 1: the drift changes softmax outputs by exactly nothing.
        p_drifted = softmax(batch.x @ weight.T + bias)
        p_undrifted = softmax((batch.x - batch.applied_shift) @ weight.T + bias)
        np.testing.assert_allclose(p_drifted, p_undrifted, atol=1e-10)

    results = [monitor.score(batch.x) for batch in post_onset]
    assert all(r.shift_alarm for r in results)
    assert all(r.direction_ratio > 0.95 for r in results)


def test_pre_onset_batches_do_not_alarm(setup):
    cfg, *_, monitor = setup
    drift = benign_drift(cfg, magnitude=4.0, onset=15)
    pre_onset = [b for b in Synthetic2DStream(cfg, drift) if b.index < 15]
    alarm_rate = np.mean([monitor.score(b.x).alarm for b in pre_onset])
    assert alarm_rate <= 0.2


def test_sensitive_drift_barely_projects_onto_blind_monitor(setup):
    cfg, *_, monitor = setup
    drift = harmful_drift(cfg, magnitude=3.0, onset=15)
    post_onset = [b for b in Synthetic2DStream(cfg, drift) if b.index >= 15]
    ratios = [monitor.score(b.x).direction_ratio for b in post_onset]
    assert np.mean(ratios) < 0.2
