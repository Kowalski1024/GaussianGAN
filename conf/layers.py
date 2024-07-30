from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Optional


@dataclass
class PointGNNConfig:
    synthethic_layers: int = 2
    use_noise: bool = False
    rank: int = 10


@dataclass
class LINKXConfig:
    synthethic_layers: int = 2
    use_noise: bool = False
    rank: int = 10


@dataclass
class GlobalPoolingConfig:
    type: str = "max"
    layers: int = 2
