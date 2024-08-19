from typing import TYPE_CHECKING

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from src.datasets.dataset_base import Dataset
import numpy as np
import PIL.Image
from pathlib import Path
import torch
from pytorch_lightning.utilities import rank_zero_only


from src.utils.pylogger import RankedLogger

if TYPE_CHECKING:
    from src.loss import GANLoss

logger = RankedLogger(__name__)


def setup_snapshot_image_grid(
    dataset: Dataset,
    use_labels: bool = True,
    max_grid_size: int = 16,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rnd = np.random.RandomState(seed)
    gh = np.clip(7680 // dataset.image_size, 4, max_grid_size)
    gw = np.clip(7680 // dataset.image_size, 4, max_grid_size)

    all_indices = list(range(len(dataset)))
    rnd.shuffle(all_indices)
    grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    # Load data.
    images, labels = zip(*[dataset[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)


def create_image_grid(
    img: np.ndarray, drange: tuple[float, float], grid_size: tuple[int, int]
) -> PIL.Image:
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    grid_w, grid_h = grid_size
    _, C, H, W = img.shape
    img = img.reshape([grid_h, grid_w, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([grid_h * H, grid_w * W, C])

    assert C == 3
    return PIL.Image.fromarray(img, "RGB")


class ImgCheckpointCallback(Callback):
    def __init__(self, dataset: Dataset, path: str, interval: int = 5):
        self.dataset = dataset
        self.images_path = Path(path)
        self.interval = interval

        self.grid_size = None
        self.valid_noise = None
        self.labels = None

        self.images_path.mkdir(parents=True, exist_ok=True)

    @rank_zero_only
    def on_fit_start(self, trainer: pl.Trainer, pl_module: "GANLoss"):
        logger.info("Creating initial images for snapshot.")
        real_path = self.images_path / "real.png"
        self.grid_size, real_images, labels = setup_snapshot_image_grid(self.dataset)
        create_image_grid(real_images, (-1, 1), grid_size=self.grid_size).save(
            real_path
        )

        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.valid_noise = torch.randn(
            self.grid_size[0] * self.grid_size[1], pl_module.noise_channels
        )

        # Save initial fake image
        fake_path = self.images_path / "fake_init.png"
        noise = self.valid_noise.to(pl_module.device)
        sphere = pl_module.sphere.to(pl_module.device)
        labels = self.labels.to(pl_module.device)

        with torch.no_grad():
            fake_imgs = pl_module.generator(noise, sphere, labels)

        create_image_grid(
            fake_imgs.cpu().numpy(), (-1, 1), grid_size=self.grid_size
        ).save(fake_path)

    @rank_zero_only
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: "GANLoss"):
        current_epoch = trainer.current_epoch + 1
        if current_epoch % self.interval == 0:
            logger.info(f"Generating images for epoch {current_epoch}")
            fake_path = self.images_path / f"fake_{current_epoch}.png"
            noise = self.valid_noise.to(pl_module.device)
            sphere = pl_module.sphere.to(pl_module.device)
            labels = self.labels.to(pl_module.device)

            with torch.no_grad():
                fake_imgs = pl_module.generator(noise, sphere, labels)

            create_image_grid(
                fake_imgs.cpu().numpy(), (-1, 1), grid_size=self.grid_size
            ).save(fake_path)
