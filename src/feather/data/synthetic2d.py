"""Tier-1 synthetic 2D drift streams with ground-truth harmfulness.

Two anisotropic Gaussian classes share a covariance matrix, so the Bayes-optimal
classifier is a known linear boundary. Drift is injected as a covariate shift of
both classes along a chosen direction:

- shifts parallel to the boundary leave every margin unchanged → benign by
  construction;
- shifts along the boundary normal push one class across the frozen boundary →
  harmful by construction.

This gives cheap, exactly-controlled streams for validating FEATHER's math
before any GPU experiment (docs/revised-plan.md §4, Tier 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Literal

import numpy as np

_DIRECTION_EPS = 1e-12

DriftKind = Literal["none", "abrupt", "gradual"]


@dataclass(frozen=True)
class Synthetic2DConfig:
    """Geometry and stream parameters for the two-Gaussian stream.

    Attributes:
        batch_size: Samples per stream batch.
        n_batches: Total batches in the stream.
        seed: Seed for the stream's random generator (reproducibility is
            mandatory, rules.md §2).
        mean0: Mean of class 0.
        mean1: Mean of class 1.
        cov: Shared 2x2 covariance matrix (row-major nested tuples).
    """

    batch_size: int
    n_batches: int
    seed: int
    mean0: tuple[float, float] = (-1.5, 0.0)
    mean1: tuple[float, float] = (1.5, 0.0)
    cov: tuple[tuple[float, float], tuple[float, float]] = ((1.0, 0.0), (0.0, 0.25))

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.n_batches <= 0:
            raise ValueError(f"n_batches must be positive, got {self.n_batches}")
        cov = np.asarray(self.cov, dtype=float)
        if cov.shape != (2, 2) or not np.allclose(cov, cov.T):
            raise ValueError("cov must be a symmetric 2x2 matrix")
        if np.any(np.linalg.eigvalsh(cov) <= 0):
            raise ValueError("cov must be positive definite")


@dataclass(frozen=True)
class DriftSpec:
    """Specification of the covariate shift injected into the stream.

    Attributes:
        kind: "none" (stationary), "abrupt" (full shift from ``onset``), or
            "gradual" (linear ramp over ``duration`` batches from ``onset``).
        direction: Shift direction in input space; normalized internally.
            Required unless ``kind == "none"``.
        magnitude: Final shift length (non-negative).
        onset: Batch index at which drift begins.
        duration: Ramp length in batches (gradual drift only).
    """

    kind: DriftKind
    direction: tuple[float, float] | None = None
    magnitude: float = 0.0
    onset: int = 0
    duration: int = 1

    def __post_init__(self) -> None:
        if self.kind not in ("none", "abrupt", "gradual"):
            raise ValueError(f"unknown drift kind: {self.kind!r}")
        if self.magnitude < 0:
            raise ValueError(f"magnitude must be non-negative, got {self.magnitude}")
        if self.onset < 0:
            raise ValueError(f"onset must be non-negative, got {self.onset}")
        if self.kind == "none":
            return
        if self.direction is None:
            raise ValueError(f"drift kind {self.kind!r} requires a direction")
        norm = float(np.linalg.norm(self.direction))
        if norm < _DIRECTION_EPS:
            raise ValueError("direction must be a non-zero vector")
        if self.kind == "gradual" and self.duration < 1:
            raise ValueError(f"duration must be >= 1, got {self.duration}")

    def unit_direction(self) -> np.ndarray:
        """Return the normalized drift direction as a length-2 array."""
        if self.direction is None:
            return np.zeros(2)
        direction = np.asarray(self.direction, dtype=float)
        return direction / np.linalg.norm(direction)

    def shift_at(self, batch_index: int) -> np.ndarray:
        """Return the shift vector applied to batch ``batch_index``."""
        if self.kind == "none" or batch_index < self.onset:
            return np.zeros(2)
        if self.kind == "abrupt":
            fraction = 1.0
        else:
            fraction = min(1.0, (batch_index - self.onset + 1) / self.duration)
        return self.magnitude * fraction * self.unit_direction()


@dataclass(frozen=True)
class StreamBatch:
    """One batch of the stream, with the ground-truth shift that produced it."""

    index: int
    x: np.ndarray
    y: np.ndarray
    applied_shift: np.ndarray = field(default_factory=lambda: np.zeros(2))


class Synthetic2DStream:
    """Iterable stream of 2D two-class batches with injected covariate drift."""

    def __init__(self, config: Synthetic2DConfig, drift: DriftSpec) -> None:
        self._config = config
        self._drift = drift
        self._means = np.stack(
            [np.asarray(config.mean0, dtype=float), np.asarray(config.mean1, dtype=float)]
        )
        self._chol = np.linalg.cholesky(np.asarray(config.cov, dtype=float))

    def __iter__(self) -> Iterator[StreamBatch]:
        rng = np.random.default_rng(self._config.seed)
        for index in range(self._config.n_batches):
            y = rng.integers(0, 2, size=self._config.batch_size)
            noise = rng.standard_normal((self._config.batch_size, 2)) @ self._chol.T
            shift = self._drift.shift_at(index)
            x = self._means[y] + noise + shift
            yield StreamBatch(index=index, x=x, y=y, applied_shift=shift)


def bayes_boundary(config: Synthetic2DConfig) -> tuple[np.ndarray, float]:
    """Return (w, b) of the Bayes-optimal linear boundary: predict 1 iff w·x + b > 0.

    With equal class priors and shared covariance, the Bayes rule is linear with
    w = cov⁻¹ (mean1 − mean0) through the midpoint of the class means.
    """
    cov = np.asarray(config.cov, dtype=float)
    mean0 = np.asarray(config.mean0, dtype=float)
    mean1 = np.asarray(config.mean1, dtype=float)
    w = np.linalg.solve(cov, mean1 - mean0)
    b = -float(w @ (mean0 + mean1) / 2.0)
    return w, b


def bayes_predict(x: np.ndarray, config: Synthetic2DConfig) -> np.ndarray:
    """Predict labels for ``x`` with the frozen Bayes-optimal boundary."""
    w, b = bayes_boundary(config)
    return (x @ w + b > 0).astype(int)


def harmful_drift(
    config: Synthetic2DConfig,
    magnitude: float,
    onset: int,
    kind: DriftKind = "abrupt",
    duration: int = 1,
) -> DriftSpec:
    """Drift along the boundary normal — degrades the frozen classifier."""
    w, _ = bayes_boundary(config)
    return DriftSpec(
        kind=kind, direction=tuple(w), magnitude=magnitude, onset=onset, duration=duration
    )


def benign_drift(
    config: Synthetic2DConfig,
    magnitude: float,
    onset: int,
    kind: DriftKind = "abrupt",
    duration: int = 1,
) -> DriftSpec:
    """Drift parallel to the boundary — provably leaves every margin unchanged."""
    w, _ = bayes_boundary(config)
    parallel = (-w[1], w[0])
    return DriftSpec(
        kind=kind, direction=parallel, magnitude=magnitude, onset=onset, duration=duration
    )
