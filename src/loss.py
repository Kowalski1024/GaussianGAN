import random
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from torchvision.transforms.functional import gaussian_blur
import torch.nn.functional as F
import numpy as np
from pytorch_lightning.core import LightningModule
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from conf.main_config import MainConfig
from src.utils.pylogger import RankedLogger
from src.utils.scheluder import GapAwareLRScheduler, LinearWarmupScheduler
import src.utils.training as training_utils
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

logger = RankedLogger(__name__)


class GANLoss(LightningModule):
    def __init__(
        self,
        stylemix_prob: float,
        blur_sigma: float,
        blur_fade_epochs: int,
        r1_gamma: float,
        generator: nn.Module,
        discriminator: nn.Module,
        dataset: Dataset,
        main_config: MainConfig,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.main_config = main_config

        # config
        self.use_stylemix = stylemix_prob
        self.blur_sigma = blur_sigma
        self.curr_blur_sigma = blur_sigma
        self.blur_fade_epochs = blur_fade_epochs
        self.r1_gamma = r1_gamma

        # sphere
        self.batch_size = main_config.dataloader.batch_size
        self.noise_channels = main_config.generator.noise_channels
        self.points = main_config.generator.points
        self.sphere = self.generate_sphere(self.points, main_config.generator.knn)

        # models
        self.generator = generator
        self.discriminator = discriminator
        self.dataset = dataset

        # metrics
        self.labels = None
        self.grid_size = self.main_config.training.image_grid_size
        self.valid_z = self.generate_noise(
            self.grid_size[0] * self.grid_size[1], self.noise_channels
        )
        self.images_path = Path(f"{self.main_config.paths.output_dir}/images")
        self.images_path.mkdir(parents=True, exist_ok=True)

    def forward(self, zs, camera):
        return self.generator(zs, self.sphere, camera)

    def run_discriminator(self, images, camera):
        blur_size = int(self.curr_blur_sigma * 3)
        blur_size += blur_size % 2 - 1
        if blur_size > 0:
            images = gaussian_blur(
                images, kernel_size=blur_size, sigma=self.curr_blur_sigma
            )

        return self.discriminator(images, camera)

    def adversarial_loss(self, logits):
        return F.softplus(logits)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        r1_gamma = self.r1_gamma if batch_idx % 4 == 0 else 0.0
        opt_g, opt_d = self.optimizers()

        real_imgs, cameras = batch
        sphere = self.sphere.to(real_imgs.device)
        cameras = cameras.to(real_imgs.device)

        noise = self.generate_noise(self.batch_size, self.noise_channels)
        noise = noise.to(real_imgs.device)

        ### Train discriminator
        opt_d.zero_grad()

        # Minimize logits for generated images
        with torch.no_grad():
            fake_imgs = self.generator(noise, sphere, cameras)

        fake_logits = self.run_discriminator(fake_imgs, cameras)
        loss_fake = self.adversarial_loss(fake_logits).mean()
        self.manual_backward(loss_fake)

        # Maximize logits for real images
        real_imgs_tmp = real_imgs.detach().requires_grad_(r1_gamma != 0.0)

        real_logits = self.run_discriminator(real_imgs_tmp, cameras)
        loss_real = self.adversarial_loss(-real_logits).mean()

        # R1 regularization
        r1_penalty = self.r1_penalty(real_logits, real_imgs_tmp, r1_gamma)
        r1_penalty = r1_penalty.mean().mul(4)
        self.manual_backward(loss_real + r1_penalty)
        opt_d.step()

        ### Train generator
        opt_g.zero_grad()

        fake_imgs = self.generator(noise, sphere, cameras)
        fake_logits = self.run_discriminator(fake_imgs, cameras)

        g_loss = self.adversarial_loss(-fake_logits)
        g_loss = g_loss.mean()
        self.manual_backward(g_loss)
        opt_g.step()

        ### Progress bar
        self.log_dict(
            {
                "d_loss": loss_real + loss_fake,
                "g_loss": g_loss,
            },
            prog_bar=True,
            logger=False,
        )

        ### Log metrics
        self.log_dict(
            {
                "epoch": self.current_epoch,
                "kimg": self._kimg,
                "generator_loss": g_loss,
                "discriminator_loss": loss_real + loss_fake,
                "real_logits": real_logits.mean(),
                "real_images": fake_logits.mean(),
                "blur_sigma": self.curr_blur_sigma,
            },
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        return {
            "g_loss": g_loss,
            "real_loss": loss_real,
            "fake_loss": loss_fake,
        }

    def generate_sphere(self, points: int, k: int) -> Data:
        sphere = training_utils.fibonacci_sphere(points)
        edge_index = knn_graph(sphere, k=k, batch=None, loop=False)
        return Data(pos=sphere, edge_index=edge_index)

    def generate_noise(
        self, batch_size: int, noise_channels: int, std: float = 1.0
    ) -> torch.Tensor:
        return torch.normal(mean=0.0, std=std, size=(batch_size, noise_channels))

    def r1_penalty(self, real_logits, real_images, r1_gamma):
        if self.r1_gamma == 0:
            return torch.tensor(0.0, device=real_images.device)

        grad_real = torch.autograd.grad(
            real_logits.sum(), real_images, create_graph=True, only_inputs=True
        )[0]
        grad_penalty = grad_real.square().sum(dim=(1, 2, 3))
        grad_penalty = grad_penalty * (r1_gamma / 2)
        return grad_penalty

    def setup(self, stage):
        if stage == "fit":
            logger.info("Exporting real images")
            image_path = self.images_path / "real.png"

            images, labels = training_utils.setup_snapshot_image_grid(
                self.dataset, grid_size=self.grid_size
            )
            real_img = training_utils.create_image_grid(
                images, [-1, 1], grid_size=self.grid_size
            )
            real_img.save(image_path)

            self.labels = torch.tensor(labels, dtype=torch.float32)

    def on_train_epoch_start(self) -> None:
        if self.current_epoch < self.blur_fade_epochs:
            decay_rate = -np.log(1e-2) / self.blur_fade_epochs
            self.curr_blur_sigma = self.blur_sigma * np.exp(
                -decay_rate * self.current_epoch
            )
        else:
            self.curr_blur_sigma = 0

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
        training_config = self.main_config.training
        opt_g = hydra.utils.instantiate(
            training_config.generator_optimizer, self.generator.parameters()
        )
        opt_d = hydra.utils.instantiate(
            training_config.discriminator_optimizer, self.discriminator.parameters()
        )

        g_scheduler = LinearWarmupScheduler(opt_g, training_config.generator_warmup)
        d_scheduler = GapAwareLRScheduler(
            opt_d, **training_config.discriminator_scheduler
        )

        return [opt_g, opt_d], [g_scheduler, d_scheduler]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            **self.main_config.dataloader,
        )

    @property
    def _kimg(self):
        return round((self.global_step * self.batch_size) / 1000, 3)
