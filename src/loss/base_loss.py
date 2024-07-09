import random
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from pytorch_lightning.core import LightningModule
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from conf.main_config import MainConfig
from src.utils.pylogger import RankedLogger
from src.utils.scheluder import GapAwareLRScheduler
import src.utils.training as training_utils

logger = RankedLogger(__name__)


class BaseLoss(LightningModule):
    def __init__(
        self,
        use_stylemix: bool,
        generator: nn.Module,
        discriminator: nn.Module,
        dataset: Dataset,
        main_config: MainConfig,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.main_config = main_config

        # config
        self.use_stylemix = use_stylemix

        # sphere
        self.batch_size = main_config.dataset.batch_size
        self.z_dim = main_config.generator.z_dim
        self.points = main_config.generator.points
        self.sphere = training_utils.fibonacci_sphere(self.points)
        self.sphere_dist = torch.cdist(self.sphere, self.sphere)

        # models
        self.dataset = dataset
        self.generator = generator
        self.discriminator = discriminator

        # metrics
        self.labels = None
        self.grid_size = self.main_config.training.image_grid_size
        valid_z = self.generate_noise(self.grid_size[0], False, self.z_dim)
        self.valid_z = valid_z.repeat(self.grid_size[1], 1, 1)
        self.images_path = Path(f"{self.main_config.paths.output_dir}/images")
        self.images_path.mkdir(parents=True, exist_ok=True)
        self.real_acc_meter = training_utils.AverageValueMeter()
        self.fake_acc_meter = training_utils.AverageValueMeter()

    def forward(self, zs, camera):
        return self.generator(zs, self.sphere, camera)

    def generate_noise(
        self, batch_size: int, use_stylemix: bool, z_dim: int, std: float = 1.0
    ) -> torch.Tensor:
        noise = torch.normal(mean=0.0, std=std, size=(batch_size, z_dim))
        noise = noise.unsqueeze(1).repeat(1, self.points, 1)

        if use_stylemix and random.random() < 0.5:
            new_noise = torch.normal(mean=0.0, std=std, size=(batch_size, z_dim))
            for i in range(batch_size):
                id = random.randint(0, self.points - 1)
                idx = torch.argsort(self.sphere_dist[id])[::1]
                cutoff = int(max(random.random(), 0.1) * self.points)
                noise[i, idx[:cutoff]] = new_noise[i]

        return noise

    def setup(self, stage):
        if stage == "fit":
            logger.info("Exporting real images")
            image_path = self.images_path / "real.png"

            images, labels = training_utils.setup_snapshot_image_grid(
                self.dataset, grid_size=self.grid_size
            )
            real_img = training_utils.create_image_grid(images, [-1, 1], grid_size=self.grid_size)
            real_img.save(image_path)

            self.labels = torch.tensor(labels, dtype=torch.float32)

    def on_train_epoch_start(self) -> None:
        if self.current_epoch % self.main_config.training.image_save_interval == 0:
            z = self.valid_z.to(self.device)
            sphere = self.sphere.to(self.device)
            camera = self.labels.to(self.device)

            image_path = Path(self.images_path / f"fake_{self.current_epoch}.png")

            with torch.no_grad():
                fake_imgs = self.generator(z, sphere, camera)
                fake_img = training_utils.create_image_grid(
                    fake_imgs.cpu().numpy(),
                    drange=(-1, 1),
                    grid_size=self.grid_size,
                )
                fake_img.save(image_path)
        return super().on_train_epoch_start()

    def configure_optimizers(self) -> tuple[Optimizer, Optimizer]:
        opt_g = hydra.utils.instantiate(
            self.main_config.generator.optimizer, self.generator.parameters()
        )
        opt_d = hydra.utils.instantiate(
            self.main_config.discriminator.optimizer, self.discriminator.parameters()
        )
        d_scheduler = GapAwareLRScheduler(
            opt_d, **self.main_config.discriminator.scheluder
        )

        return [opt_g, opt_d], d_scheduler

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.main_config.dataset.num_workers,
            pin_memory=self.main_config.dataset.pin_memory,
            shuffle=True,
        )
