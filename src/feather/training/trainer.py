"""Resumable trainer with detailed logging, checkpointing, and paper-grade outputs.

Per run (``config.run_name``) the trainer produces:

    logs/<run>.log                     full DEBUG-level log of the entire process
    outputs/<run>/config.json          exact configuration of the run
    outputs/<run>/env.json             hardware/software provenance (paper §experiments)
    outputs/<run>/metrics.csv          per-epoch metrics (loss/acc/lr/time) for plots
    outputs/<run>/summary.json         final + best accuracy, wall-clock, params
    outputs/<run>/checkpoints/epoch_XXXX.pt   resumable checkpoints (last N kept)
    outputs/<run>/checkpoints/best.pt  best-validation checkpoint (always kept)
    outputs/<run>/final_model.pt       weights + architecture meta for downstream use

Resume with ``Trainer(...).fit(resume="auto")`` (latest checkpoint of the same
run) or ``fit(resume=<path>)``. Checkpoints carry optimizer, scheduler, AMP
scaler, and all RNG states, so a resumed run continues the exact trajectory.
"""

from __future__ import annotations

import csv
import json
import logging
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from feather.training.config import TrainConfig

logger = logging.getLogger("feather.training")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (incl. CUDA) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _tuplify(obj):
    """Recursively convert lists back to tuples (random.setstate needs tuples)."""
    if isinstance(obj, (list, tuple)):
        return tuple(_tuplify(item) for item in obj)
    return obj


def _numpy_rng_to_safe(state: tuple) -> dict:
    """Convert the legacy NumPy RNG state into weights_only-loadable types."""
    name, keys, pos, has_gauss, cached = state
    return {
        "name": name,
        "keys": torch.from_numpy(np.asarray(keys, dtype=np.int64)),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached),
    }


def _numpy_rng_from_safe(state: dict) -> tuple:
    return (
        state["name"],
        state["keys"].numpy().astype(np.uint32),
        state["pos"],
        state["has_gauss"],
        state["cached_gaussian"],
    )


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or None
    except OSError:
        return None


class Trainer:
    """Trains a classifier with checkpoint/resume support and full provenance."""

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        config: TrainConfig,
    ) -> None:
        self.config = config
        self.run_dir = config.output_root / config.run_name
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config.log_root.mkdir(parents=True, exist_ok=True)
        self._setup_logging(config.log_root / f"{config.run_name}.log")

        self.device = torch.device(
            config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.amp = config.amp if config.amp is not None else self.device.type == "cuda"
        if config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True

        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer()
        self.scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs)
            if config.scheduler == "cosine"
            else None
        )
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.start_epoch = 1
        self.best_val_acc = 0.0
        self._write_provenance()

    # ------------------------------------------------------------------ setup

    def _setup_logging(self, log_path: Path) -> None:
        root = logging.getLogger("feather")
        root.setLevel(logging.DEBUG)
        if not any(
            isinstance(h, logging.FileHandler)
            and Path(getattr(h, "baseFilename", "")) == log_path.resolve()
            for h in root.handlers
        ):
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            )
            root.addHandler(file_handler)
        if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
            root.addHandler(console)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        if self.config.optimizer == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.lr,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
                nesterov=True,
            )
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    def _write_provenance(self) -> None:
        n_params = sum(p.numel() for p in self.model.parameters())
        config_dict = {
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(self.config).items()},
            "model": self.model.meta() if hasattr(self.model, "meta") else repr(type(self.model)),
            "n_parameters": n_params,
        }
        (self.run_dir / "config.json").write_text(json.dumps(config_dict, indent=2))
        env = {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": self.device.type,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_vram_gb": round(
                torch.cuda.get_device_properties(0).total_memory / 2**30, 1
            ) if torch.cuda.is_available() else None,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_commit": _git_commit(),
            "amp": self.amp,
        }
        (self.run_dir / "env.json").write_text(json.dumps(env, indent=2))
        logger.info("run '%s' on %s (%s) | params=%s | amp=%s",
                    self.config.run_name, self.device,
                    env["gpu_name"] or platform.processor(), f"{n_params:,}", self.amp)
        logger.debug("full config: %s", config_dict)
        logger.debug("environment: %s", env)

    # ------------------------------------------------------------- checkpoints

    def _checkpoint_payload(self, epoch: int) -> dict:
        return {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state": self.scaler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "rng": {
                # Only weights_only=True-loadable types (tensors + primitives).
                "python": random.getstate(),
                "numpy": _numpy_rng_to_safe(np.random.get_state()),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "model_meta": self.model.meta() if hasattr(self.model, "meta") else None,
            "torch_version": torch.__version__,
        }

    def save_checkpoint(self, epoch: int, is_best: bool) -> None:
        path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        torch.save(self._checkpoint_payload(epoch), path)
        logger.info("checkpoint saved: %s", path)
        if is_best:
            torch.save(self._checkpoint_payload(epoch), self.checkpoint_dir / "best.pt")
            logger.info("new best model (val_acc=%.4f) saved to best.pt", self.best_val_acc)
        self._prune_checkpoints()

    def _prune_checkpoints(self) -> None:
        epochs = sorted(self.checkpoint_dir.glob("epoch_*.pt"))
        for old in epochs[: -self.config.keep_last]:
            old.unlink()
            logger.debug("pruned old checkpoint %s", old.name)

    def _resolve_resume(self, resume: str | Path) -> Path:
        if str(resume) == "auto":
            candidates = sorted(self.checkpoint_dir.glob("epoch_*.pt"))
            if not candidates:
                raise FileNotFoundError(
                    f"resume='auto' but no checkpoints in {self.checkpoint_dir}"
                )
            return candidates[-1]
        path = Path(resume)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return path

    def load_checkpoint(self, resume: str | Path) -> None:
        """Restore model/optimizer/scheduler/scaler/RNG state from a checkpoint."""
        path = self._resolve_resume(resume)
        # Checkpoints are stored with weights_only-safe types, so the safe
        # loader works even though these are our own files (rules.md §3).
        payload = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        if self.scheduler and payload["scheduler_state"]:
            self.scheduler.load_state_dict(payload["scheduler_state"])
        self.scaler.load_state_dict(payload["scaler_state"])
        self.best_val_acc = payload["best_val_acc"]
        rng = payload["rng"]
        random.setstate(_tuplify(rng["python"]))
        np.random.set_state(_numpy_rng_from_safe(rng["numpy"]))
        torch.set_rng_state(rng["torch"].cpu())
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
        self.start_epoch = payload["epoch"] + 1
        logger.info(
            "resumed from %s (epoch %d done, best_val_acc=%.4f); continuing at epoch %d",
            path, payload["epoch"], self.best_val_acc, self.start_epoch,
        )

    # ------------------------------------------------------------------- loops

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(self.config.seed)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.config.num_workers > 0,
            generator=generator if shuffle else None,
        )

    def _train_epoch(self, loader: DataLoader, epoch: int) -> tuple[float, float]:
        self.model.train()
        total_loss, correct, seen = 0.0, 0, 0
        window_start, window_images = time.perf_counter(), 0
        for step, (inputs, targets) in enumerate(loader, start=1):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(self.device.type, enabled=self.amp):
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch = targets.size(0)
            total_loss += loss.item() * batch
            correct += (logits.argmax(1) == targets).sum().item()
            seen += batch
            window_images += batch
            if step % self.config.log_interval == 0:
                elapsed = time.perf_counter() - window_start
                logger.info(
                    "epoch %d step %d/%d | loss=%.4f acc=%.4f lr=%.5f | %.0f img/s",
                    epoch, step, len(loader), total_loss / seen, correct / seen,
                    self.optimizer.param_groups[0]["lr"],
                    window_images / max(elapsed, 1e-9),
                )
                window_start, window_images = time.perf_counter(), 0
        return total_loss / seen, correct / seen

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        """Return (loss, accuracy) of the model on a loader."""
        self.model.eval()
        total_loss, correct, seen = 0.0, 0, 0
        for inputs, targets in loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            with torch.amp.autocast(self.device.type, enabled=self.amp):
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)
            total_loss += loss.item() * targets.size(0)
            correct += (logits.argmax(1) == targets).sum().item()
            seen += targets.size(0)
        return total_loss / seen, correct / seen

    def _append_metrics(self, row: dict) -> None:
        path = self.run_dir / "metrics.csv"
        new_file = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    # --------------------------------------------------------------------- fit

    def fit(self, resume: str | Path | None = None) -> dict:
        """Train (or resume) to ``config.epochs``; returns the summary dict."""
        seed_everything(self.config.seed)
        if resume is not None:
            self.load_checkpoint(resume)
        train_loader = self._loader(self.train_dataset, shuffle=True)
        val_loader = self._loader(self.val_dataset, shuffle=False)
        logger.info(
            "training '%s': epochs %d..%d | train=%d val=%d | batch=%d",
            self.config.run_name, self.start_epoch, self.config.epochs,
            len(self.train_dataset), len(self.val_dataset), self.config.batch_size,
        )
        run_start = time.perf_counter()
        best_epoch = self.start_epoch - 1
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            epoch_start = time.perf_counter()
            train_loss, train_acc = self._train_epoch(train_loader, epoch)
            val_loss, val_acc = self.evaluate(val_loader)
            if self.scheduler:
                self.scheduler.step()
            epoch_seconds = time.perf_counter() - epoch_start
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                best_epoch = epoch
            logger.info(
                "epoch %d/%d done in %.1fs | train loss=%.4f acc=%.4f | "
                "val loss=%.4f acc=%.4f%s",
                epoch, self.config.epochs, epoch_seconds,
                train_loss, train_acc, val_loss, val_acc, " (best)" if is_best else "",
            )
            self._append_metrics({
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "lr": self.optimizer.param_groups[0]["lr"],
                "epoch_seconds": round(epoch_seconds, 2),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            if epoch % self.config.checkpoint_every == 0 or epoch == self.config.epochs:
                self.save_checkpoint(epoch, is_best)

        total_seconds = time.perf_counter() - run_start
        summary = {
            "run_name": self.config.run_name,
            "seed": self.config.seed,
            "epochs_trained": self.config.epochs,
            "best_val_acc": round(self.best_val_acc, 6),
            "best_epoch": best_epoch,
            "final_val_acc": round(val_acc, 6),
            "total_seconds": round(total_seconds, 1),
            "device": self.device.type,
            "amp": self.amp,
            "n_parameters": sum(p.numel() for p in self.model.parameters()),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "model_meta": self.model.meta() if hasattr(self.model, "meta") else None,
                "summary": summary,
            },
            self.run_dir / "final_model.pt",
        )
        logger.info("run complete: %s", summary)
        return summary
