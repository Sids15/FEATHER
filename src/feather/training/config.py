"""Training configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    """All knobs for one training run (serialized to outputs/<run>/config.json).

    Attributes:
        run_name: Unique name of the run; determines the output/log locations
            (reuse the same name with ``--resume auto`` to continue a run).
        epochs: Total epochs to train (resume counts toward this total).
        seed: Global random seed (Python, NumPy, PyTorch, CUDA).
        batch_size: Mini-batch size.
        optimizer: "sgd" (momentum + nesterov) or "adam".
        lr: Peak learning rate.
        momentum: SGD momentum (ignored for adam).
        weight_decay: L2 weight decay.
        scheduler: "cosine" (annealed over ``epochs``) or "none".
        amp: Mixed precision; None = auto (on for CUDA, off for CPU).
        checkpoint_every: Save a resumable checkpoint every N epochs.
        keep_last: How many epoch checkpoints to retain (best.pt always kept).
        log_interval: Log a training line every N batches.
        num_workers: DataLoader workers (set ~8-12 on the 24-core workstation).
        device: "cuda", "cpu", or None = auto-detect.
        output_root: Folder that receives outputs/<run_name>/.
        log_root: Folder that receives logs/<run_name>.log.
        deterministic: Force cuDNN determinism (slower; for debugging only —
            benchmark mode is used otherwise for speed).
    """

    run_name: str
    epochs: int
    seed: int = 0
    batch_size: int = 128
    optimizer: str = "sgd"
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    scheduler: str = "cosine"
    amp: bool | None = None
    checkpoint_every: int = 1
    keep_last: int = 3
    log_interval: int = 50
    num_workers: int = 4
    device: str | None = None
    output_root: Path = field(default_factory=lambda: Path("outputs"))
    log_root: Path = field(default_factory=lambda: Path("logs"))
    deterministic: bool = False

    def __post_init__(self) -> None:
        if not self.run_name or any(c in self.run_name for c in r'\/:*?"<>| '):
            raise ValueError(
                f"run_name must be a non-empty filesystem-safe token, got {self.run_name!r}"
            )
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.optimizer not in ("sgd", "adam"):
            raise ValueError(f"optimizer must be 'sgd' or 'adam', got {self.optimizer!r}")
        if self.scheduler not in ("cosine", "none"):
            raise ValueError(f"scheduler must be 'cosine' or 'none', got {self.scheduler!r}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.checkpoint_every <= 0:
            raise ValueError(f"checkpoint_every must be positive, got {self.checkpoint_every}")
        if self.keep_last <= 0:
            raise ValueError(f"keep_last must be positive, got {self.keep_last}")
        self.output_root = Path(self.output_root)
        self.log_root = Path(self.log_root)
