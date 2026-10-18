"""Per-batch online cost of FEATHER and of the full-space mean-shift gate.

Times only the monitors (no forward pass, no I/O) on a saved CIFAR-10 run's
calibration activations, CPU, NumPy:

    python src/experiments/monitor_cost.py outputs/monitor_cifar10_seed0
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from feather.core.monitor import MonitorConfig, SubspaceDriftMonitor


def ms_per_call(fn, repeats: int = 200) -> float:
    fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return 1000 * (time.perf_counter() - start) / repeats


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    raw = Path(sys.argv[1]) / "raw"
    # bases were saved as float32; re-orthonormalize in float64 (same subspace)
    basis, _ = np.linalg.qr(np.load(raw / "monitor.npz")["blind_basis"].astype(np.float64))
    phi = np.load(raw / "calibration.npz")["phi"].astype(np.float64)
    monitor = SubspaceDriftMonitor(basis, phi, MonitorConfig(batch_size=500))
    batch, mu_ref = phi[:500], phi.mean(axis=0)
    print(f"FEATHER score: {ms_per_call(lambda: monitor.score(batch)):.2f} ms / 500-image batch")
    print(f"mean-shift gate: {ms_per_call(lambda: np.linalg.norm(batch.mean(axis=0) - mu_ref)):.3f} ms")


if __name__ == "__main__":
    main()
