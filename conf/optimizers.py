from dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class OptimizerConfig:
    _target_: str = MISSING
    lr: float = MISSING


@dataclass
class ScheduledConfig:
    ideal_loss: float = 0.5
    x_min: float = 0.05
    x_max: float = 0.05
    h_min: float = 0.1
    f_max: float = 2.0


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
