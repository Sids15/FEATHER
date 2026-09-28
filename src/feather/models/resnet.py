"""ResNet-18 adapted for CIFAR-scale inputs (3x32x32).

Standard CIFAR modification of the torchvision ResNet-18: 3x3 stride-1 stem
instead of 7x7 stride-2, and no initial max-pool, so 32x32 inputs are not
collapsed in the first layers.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


class CifarResNet18(nn.Module):
    """ResNet-18 (CIFAR variant) with penultimate-feature access.

    Attributes:
        feature_dim: Dimension of the penultimate activation (512).
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = 512
        net = resnet18(weights=None, num_classes=num_classes)
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        self.net = net

    @property
    def head(self) -> nn.Linear:
        """The final softmax head (source of W, b for the Fisher matrix)."""
        return self.net.fc

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate activations of shape (batch, 512)."""
        n = self.net
        x = n.relu(n.bn1(n.conv1(x)))
        x = n.maxpool(x)
        x = n.layer4(n.layer3(n.layer2(n.layer1(x))))
        x = torch.flatten(n.avgpool(x), 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def meta(self) -> dict:
        """Serializable architecture description (stored with checkpoints)."""
        return {"arch": "cifar_resnet18", "num_classes": self.num_classes}


class ResNet18(nn.Module):
    """Standard ResNet-18 (unmodified 7x7 stem + max-pool) for medium images
    such as WILDS Camelyon17 (3x96x96); same ``features``/``head`` interface.

    ``pretrained`` only affects the initial weights at *training* time and is
    deliberately left out of :meth:`meta`, so :func:`load_frozen_model` rebuilds
    the architecture without re-downloading ImageNet weights before the saved
    ``state_dict`` is loaded over them.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = False) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = 512
        weights = "IMAGENET1K_V1" if pretrained else None
        net = resnet18(weights=weights)
        net.fc = nn.Linear(self.feature_dim, num_classes)
        self.net = net

    @property
    def head(self) -> nn.Linear:
        """The final softmax head (source of W, b for the Fisher matrix)."""
        return self.net.fc

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate activations of shape (batch, 512)."""
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        x = n.layer4(n.layer3(n.layer2(n.layer1(x))))
        return torch.flatten(n.avgpool(x), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def meta(self) -> dict:
        """Serializable architecture description (stored with checkpoints)."""
        return {"arch": "resnet18", "num_classes": self.num_classes}
