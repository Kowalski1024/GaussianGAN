from dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class LossConfig:
    _target_: str = MISSING
    use_stylemix: bool = True


@dataclass
class LSGANLossConfig(LossConfig):
    _target_: str = "src.loss.lsgan_loss.LSGANLoss"