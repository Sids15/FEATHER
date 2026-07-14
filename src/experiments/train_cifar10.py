"""Train the CIFAR ResNet-18 on clean CIFAR-10 (Tier-3 base model).

Typical use on the workstation (from the repo root, venv active):

    python src/experiments/train_cifar10.py --run-name cifar10_seed0 --seed 0
    python src/experiments/train_cifar10.py --run-name cifar10_seed0 --resume auto

Defaults (SGD 0.1, cosine, 50 epochs, batch 128) reach ~93-94% val accuracy
in roughly 5-10 minutes on the RTX 4500 Ada. For the multi-seed benchmark run
this once per seed (0..4). Outputs land in outputs/<run-name>/, the full log
in logs/<run-name>.log. Dataset root: --data-root or FEATHER_DATA_DIR.
"""

from __future__ import annotations

import argparse

from feather.data.vision import cifar10_datasets
from feather.models import CifarResNet18
from feather.training import TrainConfig, Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True, help="unique run id (reuse with --resume auto)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--no-augment", action="store_true", help="disable train augmentation")
    parser.add_argument("--data-root", default=None, help="overrides FEATHER_DATA_DIR")
    parser.add_argument("--device", default=None, choices=[None, "cuda", "cpu"])
    parser.add_argument("--resume", default=None, help="'auto' or a checkpoint path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_ds, val_ds = cifar10_datasets(args.data_root, augment=not args.no_augment)
    config = TrainConfig(
        run_name=args.run_name,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer="sgd",
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler="cosine",
        checkpoint_every=args.checkpoint_every,
        num_workers=args.num_workers,
        device=args.device,
    )
    trainer = Trainer(CifarResNet18(), train_ds, val_ds, config)
    trainer.fit(resume=args.resume)


if __name__ == "__main__":
    main()
