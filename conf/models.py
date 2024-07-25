from dataclasses import dataclass, field
from omegaconf import MISSING
import conf.optimizers as optimizers
from conf.layers import PointGNNConfig, GlobalPoolingConfig, LINKXConfig
from typing import Optional


@dataclass
class CloudNetworkConfig:
    in_channels: int = 128
    hidden_channels: list[int] = field(default_factory=lambda: [128, 128])
    out_channels: int = 128
    layers_config: PointGNNConfig = field(default_factory=PointGNNConfig)
    pooling_config: GlobalPoolingConfig = field(default_factory=GlobalPoolingConfig(type="max"))


@dataclass
class FeatureNetworkConfig:
    in_channels: int = 128
    hidden_channels: list[int] = field(default_factory=lambda: [256, 256, 256])
    out_channels: int = 256
    layers_config: LINKXConfig = field(default_factory=LINKXConfig)


@dataclass
class DecoderConfig:
    in_channels: int = 256
    hidden_channels: list[int] = field(default_factory=lambda: [256, 256])
    max_scale: float = 0.02
    shs_degree: int = 3
    use_rgb: bool = True
    xyz_offset: bool = True
    restrict_offset: bool = True


@dataclass
class GeneratorConfig:
    _target_: str = MISSING
    points: int = 4096
    knn: int = 6

    noise_channels: int = 512

    # mapping network
    mapping_layers: int = 8

    cloud_network: CloudNetworkConfig = field(default_factory=CloudNetworkConfig)
    feature_network: FeatureNetworkConfig = field(default_factory=FeatureNetworkConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)


    # optimizer
    optimizer: optimizers.OptimizerConfig = field(
        default_factory=optimizers.AdamWOptimizerConfig
    )



@dataclass
class DiscriminatorConfig:
    _target_: str = MISSING
    in_channels: int = 128
    hidden_channels: int = 128
    out_channels: int = 128
    style_channels: int = 128
    num_layers: int = 3
    learning_rate: float = 1.0