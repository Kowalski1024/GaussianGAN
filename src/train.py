from pytorch_lightning.core import LightningModule
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import random

from src.network.generator import ImageGenerator
from src.network.discriminator import Discriminator
from src.dataset import SyntheticDataset
from src.utils.training import save_image_grid, fibonacci_sphere
from conf.main_config import MainConfig
import hydra
from pathlib import Path


class LSGAN(LightningModule):
    def __init__(self, config: MainConfig):
        super().__init__()
        self.automatic_optimization = False
        self.config = config
        self.points = config.generator.points
        self.image_resolution = config.image_resolution
        self.batch_size = config.dataset.batch_size

        self.generator = ImageGenerator(
            config=config.generator, image_resolution=self.image_resolution
        )
        self.discriminator = Discriminator(
            config=config.discriminator, image_resolution=self.image_resolution
        )

        self.valid_z = self.generate_noise()
        self.sphere = fibonacci_sphere(self.points)

    def forward(self, zs, camera):
        return self.generator(zs, camera)

    def adversarial_loss(self, y_hat, y):
        return F.mse_loss(y_hat, y)

    @staticmethod
    def smooth_label(logits, min=0.9, max=1.0):
        return torch.rand_like(logits) * (max - min) + min

    @staticmethod
    def noisy_label(label, flip_probability=0.05):
        flip = torch.rand_like(label) < flip_probability
        inverse_label = 1 - label
        label[flip] = inverse_label[flip]

        return label

    def generate_noise(self, std=0.2, z_dim=128):
        noise = torch.normal(mean=0.0, std=std, size=(self.batch_size, z_dim))
        noise = noise.unsqueeze(1).repeat(1, self.points, 1)

        # if self.use_stylemix and random.random() < 0.5:
        #     distances = self.generator.sphere_dist

        #     new_noise = torch.normal(mean=0.0, std=std, size=(self.batch_size, z_dim))
        #     for i in range(self.batch_size):
        #         id = random.randint(0, self.points - 1)
        #         idx = torch.argsort(distances[id])[::1]
        #         cutoff = int(max(random.random(), 0.1) * self.points)
        #         noise[i, idx[:cutoff]] = new_noise[i]

        return noise

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()

        real_imgs, camera = batch

        # Train discriminator
        opt_d.zero_grad()
        noise = self.generate_noise()
        noise = noise.to(real_imgs.device)
        sphere = self.sphere.to(real_imgs.device)

        with torch.no_grad():
            fake_imgs = self.generator(noise, sphere, camera)

        real_imgs = real_imgs  # .requires_grad_(True)
        real_logits = self.discriminator(real_imgs, camera)
        fake_logits = self.discriminator(fake_imgs, camera)

        # fake_sign = F.sigmoid(fake_logits)
        # real_sign = F.sigmoid(real_logits)

        fake_label = torch.zeros_like(fake_logits)
        real_label = self.smooth_label(real_logits)
        real_label = self.noisy_label(real_label)

        fake_label, real_label = (
            fake_label.to(fake_logits.device),
            real_label.to(real_logits.device),
        )

        # r1_grads = torch.autograd.grad(
        #     real_logits.sum(), real_imgs, create_graph=True, only_inputs=True
        # )[0]

        # r1_penalty = r1_grads.square().sum(dim=(1, 2, 3))
        # loss_dr = r1_penalty * (self.gamma / 2)
        loss_fake = self.adversarial_loss(fake_logits, fake_label)
        loss_real = self.adversarial_loss(real_logits, real_label)

        d_loss = (loss_fake + loss_real) / 2  # + loss_dr.mean()
        self.manual_backward(d_loss)
        opt_d.step()

        self.log_dict(
            {
                "real_sign": F.tanh(real_logits).mean(),
                "fake_sign": F.tanh(fake_logits).mean(),
            },
            prog_bar=True,
        )
        self.log_dict(
            {
                "d_loss": d_loss,
                "fake_score": fake_logits.mean(),
                "real_score": real_logits.mean(),
            }
        )

        # Train generator
        opt_g.zero_grad()
        noise = self.generate_noise()
        noise = noise.to(real_imgs.device)

        fake_imgs = self.generator(noise, sphere, camera)
        fake_logits = self.discriminator(fake_imgs, camera)
        # fake_sign = F.sigmoid(fake_logits)

        fake_label = torch.ones_like(fake_logits)
        fake_label = self.noisy_label(fake_label)

        fake_label = fake_label.to(fake_logits.device)

        g_loss = self.adversarial_loss(fake_logits, fake_label)
        self.manual_backward(g_loss)
        opt_g.step()

        self.log("g_loss", g_loss)

        return {"d_loss": d_loss, "g_loss": g_loss}

    def on_train_epoch_start(self) -> None:
        z = self.valid_z.to(self.device)
        sphere = self.sphere.to(self.device)

        grid_size = int(self.batch_size**0.5)

        image_path = Path(
            f"{self.config.paths.output_dir}/images/fake_{self.current_epoch}.png"
        )
        image_path.parent.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            fake_imgs = self.generator(z, sphere, None)
            img = save_image_grid(
                fake_imgs[: grid_size**2].detach().cpu().numpy(),
                drange=(-1, 1),
                grid_size=(grid_size, grid_size),
            )
            img.save(image_path)
        return super().on_train_epoch_start()

    def configure_optimizers(self):
        opt_g = hydra.utils.instantiate(
            self.config.generator.optimizer, self.generator.parameters()
        )
        opt_d = hydra.utils.instantiate(
            self.config.discriminator.optimizer, self.discriminator.parameters()
        )

        return opt_g, opt_d

    def train_dataloader(self):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        dataset = SyntheticDataset(self.config.dataset.data_dir, transform)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.config.dataset.num_workers,
            pin_memory=self.config.dataset.pin_memory,
            shuffle=True,
        )
