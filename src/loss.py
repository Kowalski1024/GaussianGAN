import hydra
import numpy as np
from pytorch_lightning.core import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import gaussian_blur

from conf.main_config import MainConfig
from src.datasets import Dataset
from src.utils.pylogger import RankedLogger
from src.utils.scheluder import GapAwareLRScheduler, LinearWarmupScheduler
import src.utils.training as training_utils

logger = RankedLogger(__name__)


class GANLoss(LightningModule):
    def __init__(
        self,
        stylemix_prob: float,
        blur_sigma: float,
        blur_fade_epochs: int,
        r1_gamma: float,
        r1_interval: int,
        generator: nn.Module,
        discriminator: nn.Module,
        train_dataset: Dataset,
        test_dataset: Dataset,
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
        self.r1_interval = r1_interval

        # sphere
        self.batch_size = main_config.dataloader.batch_size
        self.noise_channels = main_config.generator.noise_channels
        self.points = main_config.generator.points
        self.sphere = training_utils.generate_sphere(self.points, main_config.generator.knn)

        # models
        self.generator = generator
        self.discriminator = discriminator
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def forward(self, zs, camera):
        return self.generator(zs, self.sphere, camera)

    def run_discriminator(self, images, camera):
        blur_size = int(self.curr_blur_sigma * 3)
        blur_size += blur_size % 2 - 1
        if blur_size > 0:
            images = gaussian_blur(images, kernel_size=blur_size, sigma=self.curr_blur_sigma)

        return self.discriminator(images, camera)

    def adversarial_loss(self, logits):
        return F.softplus(logits)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        opt_g, opt_d = self.optimizers()
        g_scheduler, d_scheduler = self.lr_schedulers()

        real_imgs, cameras = batch
        sphere = self.sphere.to(real_imgs.device)
        cameras = cameras.to(real_imgs.device)

        noise = training_utils.generate_noise(self.batch_size, self.noise_channels)
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
        real_logits = self.run_discriminator(real_imgs, cameras)
        loss_real = self.adversarial_loss(-real_logits).mean()
        self.manual_backward(loss_real)

        # Lazy R1 regularization
        if (batch_idx + 1) % self.r1_interval == 0 and self.r1_gamma != 0.0:
            real_imgs_tmp = real_imgs.clone().detach().requires_grad_(True)
            real_logits = self.run_discriminator(real_imgs_tmp, cameras)
            r1_penalty = training_utils.r1_penalty(real_logits, real_imgs_tmp, self.r1_gamma)
            r1_penalty = r1_penalty.mean().mul(self.r1_interval)
            self.manual_backward(r1_penalty)

        opt_d.step()

        ### Train generator
        opt_g.zero_grad()

        fake_imgs = self.generator(noise, sphere, cameras)
        fake_logits = self.run_discriminator(fake_imgs, cameras)

        g_loss = self.adversarial_loss(-fake_logits)
        g_loss = g_loss.mean()
        self.manual_backward(g_loss)
        opt_g.step()
        g_scheduler.step()

        ### Progress bar
        self.log_dict(
            {
                "d_loss": loss_real + loss_fake,
                "g_loss": g_loss,
                "curr_kimg": self._kimg,
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

    def on_train_epoch_start(self) -> None:
        if self.current_epoch < self.blur_fade_epochs:
            decay_rate = -np.log(1.0 / self.blur_sigma) / self.blur_fade_epochs
            self.curr_blur_sigma = self.blur_sigma * np.exp(-decay_rate * self.current_epoch)
        else:
            self.curr_blur_sigma = 0.0
        return super().on_train_epoch_start()

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        pass

    def configure_optimizers(self):
        logger.info("Configuring optimizers.")
        training_config = self.main_config.training
        g_optimizer_cfg = training_config.generator_optimizer
        d_optimizer_cfg = training_config.discriminator_optimizer

        # Adjust discriminator optimizer for R1 regularization
        if self.r1_gamma != 0:
            d_optimizer_cfg = training_utils.adjust_optimizer(d_optimizer_cfg, self.r1_interval)

        g_lr = g_optimizer_cfg.lr
        opt_g = hydra.utils.instantiate(
            g_optimizer_cfg,
            [
                {
                    "params": self.generator.gaussian_generator.mapping_network.parameters(),
                    "lr": g_lr,
                },
                {
                    "params": self.generator.gaussian_generator.decoder.parameters(),
                    "lr": g_lr * 0.1,
                },
                {
                    "params": self.generator.gaussian_generator.cloud_network.parameters(),
                    "lr": g_lr,
                },
                {
                    "params": self.generator.gaussian_generator.feature_network.parameters(),
                    "lr": g_lr,
                },
            ],
        )
        opt_d = hydra.utils.instantiate(
            d_optimizer_cfg,
            self.discriminator.parameters(),
        )

        g_scheduler = LinearWarmupScheduler(opt_g, training_config.generator_warmup)
        d_scheduler = GapAwareLRScheduler(opt_d, **training_config.discriminator_scheduler)

        return [opt_g, opt_d], [g_scheduler, d_scheduler]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            **self.main_config.dataloader,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            **self.main_config.dataloader,
        )

    @property
    def _kimg(self):
        return round((self.global_step // 2 * self.batch_size * self.trainer.world_size) / 1000, 3)
