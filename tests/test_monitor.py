"""Tests for the subspace drift monitor (statistics + bootstrap calibration).

Statistics per batch (docs/revised-plan.md §2):
- direction ratio  s = ‖P Δμ‖ / ‖Δμ‖          (where is the drift pointing?)
- shift magnitude  m = ‖P Δμ‖                   (is it big enough to matter?)
- energy           v = mean ‖P (φ − μ_ref)‖²    (catches variance-only drift)
Thresholds for m and v come from bootstrap quantiles on the reference set.
"""

import numpy as np
import pytest

from feather.core.monitor import MonitorConfig, SubspaceDriftMonitor


D = 3
BASIS = np.array([[1.0, 0.0, 0.0]]).T  # monitor the x-axis subspace
ORTHOGONAL = np.array([0.0, 1.0, 0.0])


def make_reference(n=10_000, seed=7) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, D))


def make_monitor(**overrides) -> SubspaceDriftMonitor:
    defaults = dict(batch_size=500, n_bootstrap=300, quantile=0.99, seed=11)
    defaults.update(overrides)
    return SubspaceDriftMonitor(BASIS, make_reference(), MonitorConfig(**defaults))


def fresh_batch(seed, size=500) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((size, D))


class TestCalibration:
    def test_thresholds_are_deterministic_given_seed(self):
        a, b = make_monitor(), make_monitor()
        assert a.shift_threshold == b.shift_threshold
        assert a.energy_threshold == b.energy_threshold

    def test_stationary_alarm_rate_near_nominal(self):
        monitor = make_monitor()
        alarms = [monitor.score(fresh_batch(seed)).alarm for seed in range(200)]
        assert np.mean(alarms) <= 0.08  # two 1%-quantile stats -> ~2% expected


class TestStatistics:
    def test_in_subspace_shift_fires_alarm_with_high_direction_ratio(self):
        monitor = make_monitor()
        result = monitor.score(fresh_batch(0) + 1.0 * BASIS[:, 0])
        assert result.shift_alarm
        assert result.alarm
        assert result.direction_ratio > 0.95
        assert result.shift_magnitude > monitor.shift_threshold

    def test_orthogonal_shift_gives_low_ratio_and_no_shift_alarm(self):
        monitor = make_monitor()
        result = monitor.score(fresh_batch(0) + 5.0 * ORTHOGONAL)
        assert result.direction_ratio < 0.2
        assert not result.shift_alarm

    def test_direction_ratio_bounded_in_unit_interval(self):
        monitor = make_monitor()
        for seed in range(20):
            r = monitor.score(fresh_batch(seed)).direction_ratio
            assert 0.0 <= r <= 1.0

    def test_variance_only_drift_detected_by_energy(self):
        monitor = make_monitor()
        batch = fresh_batch(0)
        batch[:, 0] *= 3.0  # mean-preserving spread along the monitored axis
        result = monitor.score(batch)
        assert result.energy_alarm
        assert result.alarm


class TestValidation:
    def test_non_orthonormal_basis_rejected(self):
        with pytest.raises(ValueError, match="orthonormal"):
            SubspaceDriftMonitor(
                2.0 * BASIS, make_reference(), MonitorConfig(batch_size=500)
            )

    def test_empty_reference_rejected(self):
        with pytest.raises(ValueError, match="reference"):
            SubspaceDriftMonitor(BASIS, np.empty((0, D)), MonitorConfig(batch_size=500))

    def test_dimension_mismatch_rejected(self):
        with pytest.raises(ValueError):
            make_monitor().score(np.zeros((500, D + 1)))

    def test_invalid_quantile_rejected(self):
        with pytest.raises(ValueError, match="quantile"):
            MonitorConfig(batch_size=500, quantile=1.5)
