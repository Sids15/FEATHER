"""Train the SmallCNN on MNIST (Tier-2 base model).

Typical use on the workstation (from the repo root, venv active):

    python src/experiments/train_mnist.py --run-name mnist_seed0 --seed 0
    python src/experiments/train_mnist.py --run-name mnist_seed0 --resume auto

Outputs land in outputs/<run-name>/, the full log in logs/<run-name>.log.
Dataset root comes from --data-root or the FEATHER_DATA_DIR environment
variable (see docs/datasets.md and docs/training.md).
"""

from __future__ import annotations

import argparse

from feather.data.vision import mnist_datasets
from feather.models import SmallCNN
from feather.training import TrainConfig, Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True, help="unique run id (reuse with --resume auto)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--data-root", default=None, help="overrides FEATHER_DATA_DIR")
    parser.add_argument("--device", default=None, choices=[None, "cuda", "cpu"])
    parser.add_argument("--resume", default=None, help="'auto' or a checkpoint path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_ds, val_ds = mnist_datasets(args.data_root)
    config = TrainConfig(
        run_name=args.run_name,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer="adam",
        lr=args.lr,
        weight_decay=0.0,
        scheduler="none",
        checkpoint_every=args.checkpoint_every,
        num_workers=args.num_workers,
        device=args.device,
    )
    trainer = Trainer(SmallCNN(), train_ds, val_ds, config)
    trainer.fit(resume=args.resume)


if __name__ == "__main__":
    main()
