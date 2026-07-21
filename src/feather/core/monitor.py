"""Streaming subspace drift monitor with bootstrap-calibrated thresholds.

Given an orthonormal basis B of the subspace to watch (typically the Fisher
blind subspace) and a clean reference activation set, the monitor computes per
batch (docs/revised-plan.md §2):

- direction ratio  s = ‖Bᵀ Δμ‖ / ‖Δμ‖   — where is the drift pointing?
- shift magnitude  m = ‖Bᵀ Δμ‖           — is it big enough to matter?
  (fixes the scale-invariance pathology of s alone)
- energy           v = mean ‖Bᵀ(φ − μ_ref)‖²  — catches variance/shape drift
  that leaves the mean unchanged

Thresholds for m and v are the q-th quantiles of their bootstrap distributions
over clean reference batches, giving direct control of the false-alarm rate
(replaces the original plan's uncalibrated sigmoid).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_ZERO_SHIFT_EPS = 1e-12
_ORTHONORMAL_ATOL = 1e-8


@dataclass(frozen=True)
class MonitorConfig:
    """Calibration parameters for the drift monitor.

    Attributes:
        batch_size: Expected streaming batch size (bootstrap batches match it).
        n_bootstrap: Number of bootstrap batches drawn from the reference set.
        quantile: Threshold quantile in (0, 1); the per-statistic false-alarm
            rate on stationary data is approximately ``1 − quantile``.
        seed: Seed for the bootstrap sampler (calibration is deterministic).
    """

    batch_size: int
    n_bootstrap: int = 500
    quantile: float = 0.99
    seed: int = 0

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.n_bootstrap <= 0:
            raise ValueError(f"n_bootstrap must be positive, got {self.n_bootstrap}")
        if not 0.0 < self.quantile < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {self.quantile}")


@dataclass(frozen=True)
class MonitorResult:
    """Statistics and alarm decisions for one streaming batch."""

    direction_ratio: float
    shift_magnitude: float
    energy: float
    shift_alarm: bool
    energy_alarm: bool

    @property
    def alarm(self) -> bool:
        """True when either calibrated statistic exceeds its threshold."""
        return self.shift_alarm or self.energy_alarm


class SubspaceDriftMonitor:
    """Monitors activation drift within a fixed subspace of activation space."""

    def __init__(
        self, basis: np.ndarray, reference: np.ndarray, config: MonitorConfig
    ) -> None:
        """Calibrate the monitor on clean reference activations.

        Args:
            basis: Orthonormal columns (d, k) spanning the monitored subspace.
            reference: Clean reference activations, shape (n, d), n ≥ 1.
            config: Calibration parameters.

        Raises:
            ValueError: If the basis is not orthonormal, the reference set is
                empty, or dimensions disagree.
        """
        basis = np.asarray(basis, dtype=float)
        reference = np.asarray(reference, dtype=float)
        if basis.ndim != 2:
            raise ValueError(f"basis must be 2-D (d, k), got shape {basis.shape}")
        gram = basis.T @ basis
        if not np.allclose(gram, np.eye(basis.shape[1]), atol=_ORTHONORMAL_ATOL):
            raise ValueError("basis columns must be orthonormal")
        if reference.ndim != 2 or reference.shape[0] == 0:
            raise ValueError("reference must be a non-empty (n, d) array")
        if reference.shape[1] != basis.shape[0]:
            raise ValueError(
                f"reference dimension {reference.shape[1]} does not match "
                f"basis dimension {basis.shape[0]}"
            )

        self._basis = basis
        self._config = config
        self._mu_ref = reference.mean(axis=0)
        self.shift_threshold, self.energy_threshold = self._calibrate(reference)
        logger.info(
            "calibrated monitor: shift_threshold=%.6g energy_threshold=%.6g",
            self.shift_threshold,
            self.energy_threshold,
        )

    @property
    def basis(self) -> np.ndarray:
        """The monitored subspace basis (d, k) — persisted for offline recalibration."""
        return self._basis

    def _statistics(self, phi: np.ndarray) -> tuple[float, float, float]:
        """Return (direction_ratio, shift_magnitude, energy) for a batch."""
        delta_mu = phi.mean(axis=0) - self._mu_ref
        projected_shift = self._basis.T @ delta_mu
        shift_magnitude = float(np.linalg.norm(projected_shift))
        total = float(np.linalg.norm(delta_mu))
        direction_ratio = shift_magnitude / total if total > _ZERO_SHIFT_EPS else 0.0
        centered = (phi - self._mu_ref) @ self._basis
        energy = float(np.mean(np.sum(centered**2, axis=1)))
        return direction_ratio, shift_magnitude, energy

    def _calibrate(self, reference: np.ndarray) -> tuple[float, float]:
        """Bootstrap the stationary distribution of (m, v) on the reference set."""
        rng = np.random.default_rng(self._config.seed)
        shifts = np.empty(self._config.n_bootstrap)
        energies = np.empty(self._config.n_bootstrap)
        n = reference.shape[0]
        for i in range(self._config.n_bootstrap):
            idx = rng.integers(0, n, size=self._config.batch_size)
            _, shifts[i], energies[i] = self._statistics(reference[idx])
        quantile = self._config.quantile
        return float(np.quantile(shifts, quantile)), float(
            np.quantile(energies, quantile)
        )

    def score(self, phi: np.ndarray) -> MonitorResult:
        """Score one streaming batch of activations.

        Args:
            phi: Batch activations, shape (batch, d).

        Returns:
            MonitorResult with the three statistics and alarm decisions.

        Raises:
            ValueError: If the batch dimension does not match the monitor.
        """
        phi = np.asarray(phi, dtype=float)
        if phi.ndim != 2 or phi.shape[1] != self._basis.shape[0]:
            raise ValueError(
                f"batch must have shape (batch, {self._basis.shape[0]}), "
                f"got {phi.shape}"
            )
        direction_ratio, shift_magnitude, energy = self._statistics(phi)
        return MonitorResult(
            direction_ratio=direction_ratio,
            shift_magnitude=shift_magnitude,
            energy=energy,
            shift_alarm=shift_magnitude > self.shift_threshold,
            energy_alarm=energy > self.energy_threshold,
        )
