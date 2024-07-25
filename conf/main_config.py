from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
import conf.optimizers as optimizers


@dataclass
class DatasetConfig:
    data_dir: str = MISSING
    batch_size: int = 16
    num_workers: int = 1
    pin_memory: bool = True


@dataclass
class GANLossConfig:
    use_stylemix: bool = True
    blur_sigma: float = 10.0
    blur_fade_epochs: int = 0
    r1_gamma: float = 0.3


@dataclass
class GeneratorConfig:
    points: int = 4096
    noise_dim: int = 256
    knn: int = 6
    xyz_mult: float = 0.75

    cloud_channels: int = 128
    cloud_layers: int = 1

    gaussian_channels: int = 256
    gaussian_layers: int = 3

    decoder_channels: int = 256
    decoder_layers: int = 2

    shs_degree: int = 3
    use_rgb: bool = True
    xyz_offset: bool = True
    restrict_offset: bool = True

    optimizer: optimizers.OptimizerConfig = field(
        default_factory=optimizers.AdamWOptimizerConfig
    )


@dataclass
class DiscriminatorConfig:
    mapping_layers: int = 8
    mappping_in_channels: int = 18
    mapping_hidden_channels: int = 128
    mapping_out_channels: int = 128

    optimizer: optimizers.OptimizerConfig = field(
        default_factory=optimizers.AdamWOptimizerConfig
    )
    scheluder: optimizers.ScheduledConfig = field(
        default_factory=optimizers.LSGANSchedulerConfig
    )


@dataclass
class TrainingConfig:
    generator_interval: int = 1
    discriminator_interval: int = 1
    generator_warmup: int = 10
    loss: GANLossConfig = field(default_factory=GANLossConfig)

    image_grid_size: tuple[int, int] = (16, 16)
    image_save_interval: int = 1


@dataclass
class MainConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"paths": "default.yaml"},
            {"hydra": "default.yaml"},
            {"logger": "csv.yaml"},
        ]
    )
    paths: dict = field(default_factory=lambda: {})
    logger: dict = field(default_factory=lambda: {})

    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    image_resolution: int = MISSING
    seed: int = 42
    task_name: str = "train"


# register the config groups
config_store = ConfigStore.instance()
config_store.store(name="main_config", node=MainConfig)

# register the optimizer config groups
config_store.store(
    group="optimizer", name="adamw", node=optimizers.AdamWOptimizerConfig
)
config_store.store(group="optimizer", name="sgd", node=optimizers.SGDOptimizerConfig)
config_store.store(group="optimizer", name="adam", node=optimizers.AdamOptimizerConfig)
