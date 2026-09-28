"""FEATHER-Lite cascade: a cheap always-on stage-1 gate that calls the FEATHER
blind-subspace monitor (stage 2) only on flagged batches.

Motivation (paper Sect. Limitations). The FEATHER projection is cheap, but on a
long clean stream it still runs every batch. The cascade runs a cheap gate up
front on a statistic that responds to *any* activation drift, and only when that
fires does it call FEATHER to sort the shift into output-visible (defer to the
confidence monitors) or blind (raise the harmful alarm). This keeps FEATHER's
precision on blind drift while running the projection on a small fraction of
batches, for a little detection latency.

Stage-1 statistic. It must respond to blind drift, so the model's confidence or
entropy will not do (the output-blindness proposition): a confidence-triggered
cascade would never call FEATHER on exactly the drift FEATHER exists to catch.
We use the full activation-space mean shift ``m_full = ||mu_batch - mu_ref||``
--- FEATHER's own shift statistic minus the blind-subspace projection, one O(d)
mean and norm per batch --- with a threshold bootstrap-calibrated on clean
reference batches, the same recipe the FEATHER monitor uses for its own
thresholds. So the cascade adds no dependency and no second calibration idea:
stage 1 watches the whole space, stage 2 watches the blind subspace.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from feather.core.monitor import SubspaceDriftMonitor


def calibrate_shift_gate(
    reference: np.ndarray,
    mu_ref: np.ndarray,
    batch_size: int,
    n_bootstrap: int = 500,
    quantile: float = 0.99,
    seed: int = 0,
) -> float:
    """Bootstrap the clean threshold for the full-space mean-shift gate.

    Draws ``n_bootstrap`` batches of ``batch_size`` from the clean reference set
    and returns the ``quantile`` of their mean-shift magnitudes, so the gate's
    per-batch false-flag rate on stationary data is about ``1 - quantile``.
    """
    reference = np.asarray(reference, dtype=float)
    mu_ref = np.asarray(mu_ref, dtype=float)
    rng = np.random.default_rng(seed)
    n = reference.shape[0]
    shifts = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=batch_size)
        shifts[i] = np.linalg.norm(reference[idx].mean(axis=0) - mu_ref)
    return float(np.quantile(shifts, quantile))


@dataclass(frozen=True)
class CascadeRecord:
    """Per-batch outcome of the cascade."""

    batch: int
    stage1_shift: float  # full-space mean-shift magnitude this batch
    stage1_flag: bool  # stage-1 gate fired
    feather_ran: bool  # stage-2 (FEATHER) was invoked this batch
    feather_alarm: bool  # stage-2 blind alarm (meaningful only if feather_ran)
    cascade_alarm: bool  # final decision: flagged AND blind


class CascadeMonitor:
    """Two-stage monitor: cheap full-space gate, FEATHER on flagged batches."""

    def __init__(
        self,
        feather: SubspaceDriftMonitor,
        mu_ref: np.ndarray,
        shift_threshold: float,
    ) -> None:
        """Wire the stage-1 gate to a fitted FEATHER monitor.

        Args:
            feather: A calibrated FEATHER blind-subspace monitor (stage 2).
            mu_ref: Clean reference activation mean (d,).
            shift_threshold: Full-space gate threshold from
                :func:`calibrate_shift_gate`.
        """
        self._feather = feather
        self._mu_ref = np.asarray(mu_ref, dtype=float)
        if self._mu_ref.ndim != 1:
            raise ValueError(f"mu_ref must be 1-D (d,), got shape {self._mu_ref.shape}")
        self._threshold = float(shift_threshold)

    def process_batch(self, phi: np.ndarray, batch: int = 0) -> CascadeRecord:
        """Process one streaming batch of activations."""
        phi = np.asarray(phi, dtype=float)
        if phi.ndim != 2 or phi.shape[1] != self._mu_ref.shape[0]:
            raise ValueError(
                f"batch must have shape (batch, {self._mu_ref.shape[0]}), "
                f"got {phi.shape}"
            )
        shift = float(np.linalg.norm(phi.mean(axis=0) - self._mu_ref))
        flagged = shift > self._threshold
        feather_alarm = bool(self._feather.score(phi).alarm) if flagged else False
        return CascadeRecord(
            batch=batch,
            stage1_shift=round(shift, 6),
            stage1_flag=flagged,
            feather_ran=flagged,
            feather_alarm=feather_alarm,
            cascade_alarm=flagged and feather_alarm,
        )


def run_cascade(
    feather: SubspaceDriftMonitor,
    mu_ref: np.ndarray,
    shift_threshold: float,
    batches: list[np.ndarray],
) -> list[CascadeRecord]:
    """Run the cascade over a stream of activation batches (one episode)."""
    monitor = CascadeMonitor(feather, mu_ref, shift_threshold)
    return [monitor.process_batch(phi, batch=i) for i, phi in enumerate(batches)]


def summarize(
    records: list[CascadeRecord], onset: int | None = None
) -> dict[str, float | int | None]:
    """Aggregate cascade records for one episode.

    Args:
        records: Per-batch records from :func:`run_cascade`.
        onset: Batch index where drift begins (``None`` for a clean episode). The
            detection delay is the number of batches from ``onset`` to the first
            stage-1 flag at or after it, or ``None`` if never flagged.
    """
    n = len(records)
    feather_run_rate = sum(r.feather_ran for r in records) / n if n else 0.0
    alarm_rate = sum(r.cascade_alarm for r in records) / n if n else 0.0
    delay: int | None = None
    if onset is not None:
        for r in records:
            if r.batch >= onset and r.stage1_flag:
                delay = r.batch - onset
                break
    return {
        "n_batches": n,
        "feather_run_rate": round(feather_run_rate, 4),
        "cascade_alarm_rate": round(alarm_rate, 4),
        "detection_delay": delay,
    }
