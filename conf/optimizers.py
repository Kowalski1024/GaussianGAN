from dataclasses import dataclass
from omegaconf import MISSING
import numpy as np


@dataclass
class OptimizerConfig:
    _target_: str = MISSING
    lr: float = MISSING


@dataclass
class SGDOptimizerConfig(OptimizerConfig):
    _target_: str = "torch.optim.SGD"


@dataclass
class AdamOptimizerConfig(OptimizerConfig):
    _target_: str = "torch.optim.Adam"
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.5, 0.99)


@dataclass
class AdamWOptimizerConfig(OptimizerConfig):
    _target_: str = "torch.optim.AdamW"
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.5, 0.99)


@dataclass
class ScheduledConfig:
    ideal_loss: float = MISSING
    x_min: float = MISSING
    x_max: float = MISSING
    h_min: float = MISSING
    f_max: float = MISSING


@dataclass
class GANSchedulerConfig(ScheduledConfig):
    ideal_loss: float = 1.386 # np.log(4) = 1.386
    x_min: float = 0.1 * 1.386
    x_max: float = 0.1 * 1.386
    h_min: float = 0.1
    f_max: float = 2.0
