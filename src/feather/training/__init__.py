"""Training infrastructure: config, resumable trainer, checkpointing."""

from feather.training.config import TrainConfig
from feather.training.trainer import Trainer

__all__ = ["TrainConfig", "Trainer"]
