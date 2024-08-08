from dataclasses import dataclass, field


@dataclass
class PointGNNConfig:
    synthethic_layers: int = 2
    use_noise: bool = False
    rank: int = 10
    aggr: str = "max"


@dataclass
class LINKXConfig:
    synthethic_layers: int = 2
    use_noise: bool = False
    rank: int = 10


@dataclass
class GlobalPoolingConfig:
    type: str = "max"
    layers: int = 2



@dataclass
class CloudNetworkConfig:
    in_channels: int = 128
    hidden_channels: list[int] = field(default_factory=lambda: [128, 128])
    out_channels: int = 128
    layers_config: PointGNNConfig = field(default_factory=PointGNNConfig)
    pooling_config: GlobalPoolingConfig = field(default_factory=GlobalPoolingConfig)


@dataclass
class FeatureNetworkConfig:
    in_channels: int = 256
    hidden_channels: list[int] = field(default_factory=lambda: [256, 256, 256])
    out_channels: int = 256
    layers_config: LINKXConfig = field(default_factory=LINKXConfig)


@dataclass
class DecoderConfig:
    in_channels: int = 512
    hidden_channels: list[int] = field(default_factory=lambda: [256, 256])
    max_scale: float = 0.02
    shs_degree: int = 3
    use_rgb: bool = True
    xyz_offset: bool = True
    restrict_offset: bool = True


@dataclass
class GeneratorConfig:
    points: int = 8192 * 2
    knn: int = 8

    noise_channels: int = 512

    cloud_network: CloudNetworkConfig = field(default_factory=CloudNetworkConfig)
    feature_network: FeatureNetworkConfig = field(default_factory=FeatureNetworkConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

