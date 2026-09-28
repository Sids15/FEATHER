"""Train a ResNet-18 on WILDS Camelyon17 (in-distribution split) for the
natural-shift monitoring experiment.

The model is the standard-stem ResNet-18 (ImageNet-initialised by default, which
makes a few epochs enough), trained on the in-distribution ``train`` hospitals
and validated on ``id_val``. It saves ``outputs/<run-name>/final_model.pt`` in
the same format as the CIFAR/MNIST models, so ``run_monitoring.py`` and
``cascade.py`` load it unchanged.

Typical use on the workstation (after 'pip install wilds' and the ~10 GB
download, see docs/datasets.md):

    python src/experiments/train_wilds.py --run-name wilds_seed0 --seed 0

Camelyon17 train is large (~300k patches); with a pretrained backbone a handful
of epochs reaches high accuracy. Run once per seed (0..4) for the spread.
"""

from __future__ import annotations

import argparse

from feather.data.wilds import camelyon17_split
from feather.models import ResNet18
from feather.training import TrainConfig, Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True, help="unique run id (reuse with --resume auto)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--from-scratch", action="store_true",
                        help="train the backbone from random init (default: ImageNet)")
    parser.add_argument("--data-root", default=None, help="overrides FEATHER_DATA_DIR")
    parser.add_argument("--device", default=None, choices=[None, "cuda", "cpu"])
    parser.add_argument("--resume", default=None, help="'auto' or a checkpoint path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_ds = camelyon17_split(args.data_root, "train")
    val_ds = camelyon17_split(args.data_root, "id_val")
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
    model = ResNet18(num_classes=2, pretrained=not args.from_scratch)
    trainer = Trainer(model, train_ds, val_ds, config)
    trainer.fit(resume=args.resume)


if __name__ == "__main__":
    main()
