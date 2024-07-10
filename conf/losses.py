from dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class LossConfig:
    _target_: str = MISSING
    use_stylemix: bool = True
    blur_sigma: float = 10.0
    blur_fade_epochs: int = 100
    r1_gamma: float = 0.3


@dataclass
class LSGANLossConfig(LossConfig):
    _target_: str = "src.loss.lsgan_loss.LSGANLoss"