"""Small CNN for MNIST-scale inputs (1x28x28)."""

from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    """Two conv blocks + a feature layer; ~99% on MNIST in a few epochs.

    Attributes:
        feature_dim: Dimension of the penultimate activation (default 128).
    """

    def __init__(self, num_classes: int = 10, feature_dim: int = 128) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, feature_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(feature_dim, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate activations of shape (batch, feature_dim)."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def meta(self) -> dict:
        """Serializable architecture description (stored with checkpoints)."""
        return {
            "arch": "small_cnn",
            "num_classes": self.num_classes,
            "feature_dim": self.feature_dim,
        }
