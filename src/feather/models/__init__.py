"""Model architectures. Every model exposes ``features(x)`` (penultimate
activations) and ``head`` (the final Linear layer) so FEATHER's monitor can
read activations and the softmax head's (W, b) without surgery."""

from feather.models.cnn import SmallCNN
from feather.models.resnet import CifarResNet18

MODELS = {"small_cnn": SmallCNN, "cifar_resnet18": CifarResNet18}

__all__ = ["SmallCNN", "CifarResNet18", "MODELS"]
