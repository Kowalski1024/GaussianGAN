import torch
import torch.nn as nn
import torch.nn.functional as F
from src.dataset import SyntheticDataset
from src.utils.pylogger import RankedLogger
from conf.main_config import MainConfig
from src.loss.base_loss import BaseLoss

logger = RankedLogger(__name__)


def smooth_label(logits, min=0.9, max=1.0):
    return torch.rand_like(logits) * (max - min) + min


def noisy_label(label, flip_probability=0.05):
    flip = torch.rand_like(label) < flip_probability
    label[flip] = (1 - label)[flip]
    return label


class LSGANLoss(BaseLoss):
    def __init__(
        self,
        use_stylemix: bool,
        generator: nn.Module,
        discriminator: nn.Module,
        dataset: SyntheticDataset,
        main_config: MainConfig,
    ):
        super().__init__(use_stylemix, generator, discriminator, dataset, main_config)

    def adversarial_loss(self, pred, target):
        return F.mse_loss(pred, target)

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        scheluder = self.lr_schedulers()

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

            fake_label = torch.zeros_like(fake_logits)
            real_label = smooth_label(real_logits)
            real_label = noisy_label(real_label)

            fake_label, real_label = (
                fake_label.to(fake_logits.device),
                real_label.to(real_logits.device),
            )

            loss_fake = self.adversarial_loss(fake_logits, fake_label)
            loss_real = self.adversarial_loss(real_logits, real_label)

            d_loss = (loss_fake + loss_real) # / 2.0
            d_loss = d_loss.mul(gain)
            self.manual_backward(d_loss.mean())
            opt_d.step()
            lr_mult = scheluder.step(d_loss.item())

            self.real_score_meter.update(real_logits.mean())
            self.fake_score_meter.update(fake_logits.mean())

            self.log_dict(
                {
                    "real_avg_score": self.real_score_meter.avg,
                    "fake_avg_score": self.fake_score_meter.avg,
                },
                prog_bar=True,
            )
            self.log_dict(
                {
                    "d_loss": d_loss,
                    "lr_mult": lr_mult,
                    "fake_score": fake_logits.mean(),
                    "real_score": real_logits.mean(),
                }
            )

        # Train generator
        g_loss = None
        if batch_idx % (gain := self.main_config.training.generator_interval) == 0:
            opt_g.zero_grad()

            fake_imgs = self.generator(noise, sphere, camera)
            fake_logits = self.discriminator(fake_imgs, camera)

            fake_label = torch.ones_like(fake_logits)
            fake_label = noisy_label(fake_label)

            fake_label = fake_label.to(fake_logits.device)

            g_loss = self.adversarial_loss(fake_logits, fake_label)
            g_loss = g_loss.mul(gain)
            self.manual_backward(g_loss)
            opt_g.step()

            self.log("g_loss", g_loss)

        return {"d_loss": d_loss, "g_loss": g_loss}
