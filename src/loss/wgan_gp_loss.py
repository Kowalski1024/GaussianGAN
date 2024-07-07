import torch
import torch.nn as nn
import torch.nn.functional as F
from src.dataset import SyntheticDataset
from src.utils.pylogger import RankedLogger
from conf.main_config import MainConfig
from src.loss.base_loss import BaseLoss

logger = RankedLogger(__name__)


class WGANGPLoss(BaseLoss):
    def __init__(
        self,
        use_stylemix: bool,
        generator: nn.Module,
        discriminator: nn.Module,
        dataset: SyntheticDataset,
        main_config: MainConfig,
    ):
        super().__init__(use_stylemix, generator, discriminator, dataset, main_config)

    def adversarial_loss(self, pred):
        return -torch.mean(pred)

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()

        real_imgs, camera = batch
        sphere = self.sphere.to(real_imgs.device)
        noise = self.generate_noise(self.batch_size, self.use_stylemix, self.z_dim)
        noise = noise.to(real_imgs.device)

        # Train discriminator
        d_loss = None
        if batch_idx % (gain := self.main_config.training.discriminator_interval) == 0:
            opt_d.zero_grad()

            with torch.no_grad():
                fake_imgs = self.generator(noise, sphere, camera)

            real_logits = self.discriminator(real_imgs, camera)
            fake_logits = self.discriminator(fake_imgs, camera)

            gradient_penalty = self.compute_gradient_penalty(
                real_imgs, fake_imgs, camera
            )
            d_loss = (
                self.adversarial_loss(fake_logits)
                - self.adversarial_loss(real_logits)
                + 10 * gradient_penalty
            )
            d_loss.backward()
            opt_d.step()

        # Train generator
        g_loss = None
        if batch_idx % gain == 0:
            opt_g.zero_grad()

            fake_imgs = self.generator(noise, sphere, camera)
            fake_logits = self.discriminator(fake_imgs, camera)

            g_loss = self.adversarial_loss(fake_logits)
            g_loss.backward()
            opt_g.step()

        return {"d_loss": d_loss, "g_loss": g_loss}

    def compute_gradient_penalty(self, real_imgs, fake_imgs, camera):
        alpha = torch.rand(real_imgs.size(0), 1, 1, 1).to(real_imgs.device)
        interpolates = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(
            True
        )
        disc_interpolates = self.discriminator(interpolates, camera)
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
