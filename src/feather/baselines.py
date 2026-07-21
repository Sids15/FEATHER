"""Output-based drift baselines, calibrated like the subspace monitors.

Each baseline reduces a batch to the mean of a per-sample scalar computed
from the softmax probabilities, and alarms when that mean crosses a
bootstrap-calibrated quantile of its clean distribution — the same protocol
as :class:`feather.core.monitor.SubspaceDriftMonitor`, so false-alarm rates
are comparable at matched calibration.

Baselines (names are the CSV column prefixes):

- ``atc``     : ATC-style. A confidence threshold t is fit offline so that
  the fraction of calibration samples with confidence above t equals the
  calibration accuracy (Garg et al., ICLR 2022); the batch score is that
  fraction online, an accuracy estimate. Alarm on drop.
- ``conf``    : DoC-style mean max-softmax confidence (Guillory et al.,
  ICCV 2021, reduced to the confidence-difference signal). Alarm on drop.
- ``entropy`` : mean prediction entropy. Alarm on rise.
- ``proxy``   : error-proxy inspired by Amoukou et al. (2024) — a logistic
  model predicting per-sample error from (confidence, entropy, margin), fit
  on the labeled calibration split; batch score is the mean predicted error
  probability. Alarm on rise. Not a reproduction of their sequential test.

Everything here is NumPy-only: fitting consumes (probabilities, labels)
arrays, so it runs and is tested without torch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from feather.core.monitor import MonitorConfig

logger = logging.getLogger(__name__)

_EPS = 1e-12


@dataclass(frozen=True)
class ScalarBaseline:
    """A calibrated mean-of-per-sample-scores drift baseline.

    Attributes:
        name: Column prefix (``atc``, ``conf``, ``entropy``, ``proxy``).
        direction: ``"low"`` alarms when the batch mean falls below the
            threshold, ``"high"`` when it rises above.
        threshold: Bootstrap quantile of clean calibration batch means.
        reference_mean: Mean per-sample score on the calibration split.
        params: Fitted baseline-specific parameters (e.g. the ATC confidence
            threshold, the proxy weights and feature normalization).
    """

    name: str
    direction: str
    threshold: float
    reference_mean: float
    params: dict = field(default_factory=dict)

    def scores(self, probabilities: np.ndarray) -> np.ndarray:
        """Per-sample scalar scores for a batch of softmax probabilities."""
        return _SCORE_FUNCTIONS[self.name](probabilities, self.params)

    def score_batch(self, probabilities: np.ndarray) -> tuple[float, bool]:
        """Return (batch mean score, alarm) for one batch."""
        mean = float(self.scores(probabilities).mean())
        if self.direction == "low":
            return mean, mean < self.threshold
        return mean, mean > self.threshold


def _confidence(probabilities: np.ndarray) -> np.ndarray:
    return probabilities.max(axis=1)


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    return -(probabilities * np.log(probabilities + _EPS)).sum(axis=1)


def _margin(probabilities: np.ndarray) -> np.ndarray:
    top2 = np.sort(probabilities, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def _proxy_features(probabilities: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [_confidence(probabilities), _entropy(probabilities), _margin(probabilities)]
    )


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def fit_atc(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """ATC-style confidence threshold: fraction above it matches accuracy."""
    accuracy = float((probabilities.argmax(axis=1) == labels).mean())
    return float(np.quantile(_confidence(probabilities), 1.0 - accuracy))


def fit_error_proxy(
    probabilities: np.ndarray,
    labels: np.ndarray,
    l2: float = 1e-4,
    learning_rate: float = 0.5,
    n_iterations: int = 500,
) -> dict:
    """Logistic error model on (confidence, entropy, margin), NumPy only.

    Deterministic full-batch gradient descent on standardized features;
    three weights and a bias, so convergence is not a concern.
    """
    features = _proxy_features(probabilities)
    mean, std = features.mean(axis=0), features.std(axis=0) + _EPS
    x = (features - mean) / std
    y = (probabilities.argmax(axis=1) != labels).astype(float)

    weights = np.zeros(x.shape[1])
    bias = float(np.log((y.mean() + _EPS) / (1.0 - y.mean() + _EPS)))
    for _ in range(n_iterations):
        error = _sigmoid(x @ weights + bias) - y
        weights -= learning_rate * (x.T @ error / len(y) + l2 * weights)
        bias -= learning_rate * float(error.mean())
    return {
        "weights": weights.tolist(),
        "bias": bias,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
    }


def _proxy_scores(probabilities: np.ndarray, params: dict) -> np.ndarray:
    x = (_proxy_features(probabilities) - np.asarray(params["feature_mean"])) / (
        np.asarray(params["feature_std"])
    )
    return _sigmoid(x @ np.asarray(params["weights"]) + params["bias"])


_SCORE_FUNCTIONS = {
    "atc": lambda p, params: (
        _confidence(p) > params["score_threshold"]
    ).astype(float),
    "conf": lambda p, params: _confidence(p),
    "entropy": lambda p, params: _entropy(p),
    "proxy": _proxy_scores,
}

_DIRECTIONS = {"atc": "low", "conf": "low", "entropy": "high", "proxy": "high"}


def _bootstrap_threshold(
    values: np.ndarray, config: MonitorConfig, direction: str
) -> float:
    """Quantile of bootstrap clean-batch means, on the alarming side."""
    rng = np.random.default_rng(config.seed)
    n = len(values)
    means = np.empty(config.n_bootstrap)
    for i in range(config.n_bootstrap):
        means[i] = values[rng.integers(0, n, size=config.batch_size)].mean()
    quantile = 1.0 - config.quantile if direction == "low" else config.quantile
    return float(np.quantile(means, quantile))


def fit_output_baselines(
    probabilities: np.ndarray,
    labels: np.ndarray,
    config: MonitorConfig,
) -> dict[str, ScalarBaseline]:
    """Fit all four output baselines on the labeled calibration split."""
    params_by_name: dict[str, dict] = {
        "atc": {"score_threshold": fit_atc(probabilities, labels)},
        "conf": {},
        "entropy": {},
        "proxy": fit_error_proxy(probabilities, labels),
    }
    baselines = {}
    for name, params in params_by_name.items():
        values = _SCORE_FUNCTIONS[name](probabilities, params)
        direction = _DIRECTIONS[name]
        baselines[name] = ScalarBaseline(
            name=name,
            direction=direction,
            threshold=_bootstrap_threshold(values, config, direction),
            reference_mean=float(values.mean()),
            params=params,
        )
        logger.info(
            "baseline %-7s: reference mean=%.4f threshold=%.4f (%s)",
            name, baselines[name].reference_mean, baselines[name].threshold,
            direction,
        )
    return baselines


def score_output_baselines(
    baselines: dict[str, ScalarBaseline], probabilities: np.ndarray
) -> dict:
    """Score one batch against every baseline; returns CSV-ready columns."""
    record = {}
    for name, baseline in baselines.items():
        mean, alarm = baseline.score_batch(probabilities)
        record[f"{name}_score"] = round(mean, 6)
        record[f"{name}_alarm"] = int(alarm)
    return record
