from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore

import conf.models as models
import conf.optimizers as optimizers


@dataclass
class DatasetConfig:
    _target_: str = "src.datasets.snr_dataset.SNRDataset"
    path: str = "/mnt/d/Tomasz/Pulpit/GaussianGAN/datasets/cars"
    background: tuple[int, int, int] = (1, 1, 1)
    use_labels: bool = True


@dataclass
class DataloaderConfig:
    batch_size: int = 16
    num_workers: int = 1
    pin_memory: bool = True
    shuffle: bool = True


@dataclass
class GANLossConfig:
    stylemix_prob: float = 0.0
    blur_sigma: float = 10.0
    blur_fade_epochs: int = 100
    r1_gamma: float = 5.0
    r1_interval: int = 16
    scale_lambda: float = 0.5
    opacity_lambda: float = 0.001


@dataclass
class TrainingConfig:
    max_epochs: int = 10000
    limit_train_batches: int = 64
    log_every_n_steps: int = 16
    save_img_every_n_epoch: int = 10
    metric_every_n_epoch: int = 200

    generator_warmup: int = 0
    loss: GANLossConfig = field(default_factory=GANLossConfig)

    generator_optimizer: optimizers.OptimizerConfig = field(
        default_factory=optimizers.AdamWOptimizerConfig
    )
    discriminator_optimizer: optimizers.OptimizerConfig = field(
        default_factory=optimizers.AdamWOptimizerConfig
    )

    discriminator_scheduler: optimizers.ScheduledConfig = field(
        default_factory=optimizers.GANSchedulerConfig
    )


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

    generator: models.GeneratorConfig = field(default_factory=models.GeneratorConfig)
    discriminator: models.DiscriminatorConfig = field(default_factory=models.DiscriminatorConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    seed: int = 42
    enable_progress_bar: bool = False
    task_name: str = "train"


# register the config groups
config_store = ConfigStore.instance()
config_store.store(name="main_config", node=MainConfig)

# register the optimizer config groups
config_store.store(group="optimizer", name="adamw", node=optimizers.AdamWOptimizerConfig)
config_store.store(group="optimizer", name="sgd", node=optimizers.SGDOptimizerConfig)
config_store.store(group="optimizer", name="adam", node=optimizers.AdamOptimizerConfig)
