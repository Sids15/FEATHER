"""Tests for the output-based baselines (NumPy only, no torch needed)."""

import numpy as np
import pytest

from feather.baselines import (
    ScalarBaseline,
    fit_atc,
    fit_error_proxy,
    fit_output_baselines,
    score_output_baselines,
)
from feather.core.monitor import MonitorConfig

CONFIG = MonitorConfig(batch_size=100, n_bootstrap=100, quantile=0.99, seed=0)


def synthetic_probabilities(n=2000, n_classes=10, sharpness=6.0, seed=0):
    """Softmax probabilities where high confidence correlates with correctness."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, size=n)
    logits = rng.normal(0.0, 1.0, size=(n, n_classes))
    per_sample_sharpness = rng.uniform(0.5, sharpness, size=n)
    correct = rng.random(n) < (per_sample_sharpness / sharpness)
    target = np.where(correct, labels, rng.integers(0, n_classes, size=n))
    logits[np.arange(n), target] += per_sample_sharpness
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True), labels


class TestFitATC:
    def test_fraction_above_threshold_matches_accuracy(self):
        probabilities, labels = synthetic_probabilities()
        threshold = fit_atc(probabilities, labels)
        accuracy = (probabilities.argmax(axis=1) == labels).mean()
        fraction = (probabilities.max(axis=1) > threshold).mean()
        assert abs(fraction - accuracy) < 0.02

    def test_low_accuracy_gives_high_threshold_coverage(self):
        # all predictions wrong -> accuracy 0 -> threshold at the top
        probabilities, _ = synthetic_probabilities(n=500)
        wrong = (probabilities.argmax(axis=1) + 1) % 10
        threshold = fit_atc(probabilities, wrong)
        assert (probabilities.max(axis=1) > threshold).mean() <= 0.01


class TestErrorProxy:
    def test_predicts_errors_better_than_chance(self):
        probabilities, labels = synthetic_probabilities()
        params = fit_error_proxy(probabilities, labels)
        baseline = ScalarBaseline("proxy", "high", 0.0, 0.0, params)
        scores = baseline.scores(probabilities)
        errors = probabilities.argmax(axis=1) != labels
        assert scores[errors].mean() > scores[~errors].mean() + 0.1

    def test_deterministic(self):
        probabilities, labels = synthetic_probabilities()
        first = fit_error_proxy(probabilities, labels)
        second = fit_error_proxy(probabilities, labels)
        assert first["weights"] == second["weights"]


@pytest.fixture(scope="module")
def fitted():
    probabilities, labels = synthetic_probabilities()
    return fit_output_baselines(probabilities, labels, CONFIG), probabilities


class TestFitOutputBaselines:
    def test_all_four_present_with_directions(self, fitted):
        baselines, _ = fitted
        assert set(baselines) == {"atc", "conf", "entropy", "proxy"}
        assert baselines["atc"].direction == "low"
        assert baselines["entropy"].direction == "high"

    def test_clean_batches_rarely_alarm(self, fitted):
        baselines, probabilities = fitted
        rng = np.random.default_rng(1)
        alarms = 0
        for _ in range(50):
            batch = probabilities[rng.integers(0, len(probabilities), size=100)]
            record = score_output_baselines(baselines, batch)
            alarms += sum(record[f"{n}_alarm"] for n in baselines)
        # 50 batches x 4 baselines at ~1% nominal each
        assert alarms <= 20

    def test_degraded_batch_alarms(self, fitted):
        baselines, _ = fitted
        # near-uniform probabilities: confidence collapses, entropy spikes
        degraded = np.full((100, 10), 0.1) + np.linspace(0, 0.004, 10)
        degraded /= degraded.sum(axis=1, keepdims=True)
        record = score_output_baselines(baselines, degraded)
        assert record["conf_alarm"] and record["entropy_alarm"]
        assert record["atc_alarm"] and record["proxy_alarm"]

    def test_score_columns_shape(self, fitted):
        baselines, probabilities = fitted
        record = score_output_baselines(baselines, probabilities[:100])
        assert set(record) == {
            f"{n}_{kind}" for n in baselines for kind in ("score", "alarm")
        }


class TestBootstrapDeterminism:
    def test_same_seed_same_thresholds(self):
        probabilities, labels = synthetic_probabilities()
        a = fit_output_baselines(probabilities, labels, CONFIG)
        b = fit_output_baselines(probabilities, labels, CONFIG)
        assert all(a[n].threshold == b[n].threshold for n in a)
