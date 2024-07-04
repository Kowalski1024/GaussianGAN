from dataclasses import dataclass
from omegaconf import MISSING


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
