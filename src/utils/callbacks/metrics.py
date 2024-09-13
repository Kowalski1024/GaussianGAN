from pathlib import Path
from typing import TYPE_CHECKING

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
import torch
from torchmetrics import MetricCollection
from torchmetrics.image import FrechetInceptionDistance, KernelInceptionDistance

from src.utils.pylogger import RankedLogger
import src.utils.training as training_utils

if TYPE_CHECKING:
    from src.loss import GANLoss

logger = RankedLogger(__name__)


class DetectorWapper(torch.nn.Module):
    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    def forward(self, x):
        return self.detector(x, return_features=True)


class MetricsCallback(Callback):
    def __init__(
        self,
        selected_metrics: list[str],
        cached_features_path: str,
        interval: int,
        num_samples: int | None = None,
        recalculate: bool = False,
    ):
        self.interval = interval
        self.num_samples = num_samples
        self.cache_path = Path(cached_features_path)
        self.recalculate = recalculate

        self.inception_model = DetectorWapper(training_utils.get_inception_model())

        metrics = []
        for metric in selected_metrics:
            match metric:
                case "fid":
                    metrics.append(
                        FrechetInceptionDistance(
                            feature=self.inception_model, reset_real_features=False
                        )
                    )
                case "kid":
                    metrics.append(
                        KernelInceptionDistance(
                            feature=self.inception_model, reset_real_features=False
                        )
                    )

        self.metrics = MetricCollection(metrics)
        self.metrics.persistent(True)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: "GANLoss"):
        device = pl_module.device
        self.metrics.to(device)

        # calculate real metrics
        dataloader = pl_module.val_dataloader()
        self.calculate_real_metrics(dataloader, device)

        self.num_samples = self.num_samples or len(dataloader.dataset)

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: "GANLoss"):
        current_epoch = trainer.current_epoch + 1
        if current_epoch % self.interval == 0:
            logger.info("Calculating metrics...")
            generator = pl_module.generator
            noise_channels = pl_module.noise_channels
            init_points = pl_module.sphere
            device = pl_module.device
            dataloader = pl_module.val_dataloader()
            batchsize = pl_module.batch_size
            label_iter = training_utils.label_iterator(dataloader.dataset, batchsize)

            # calculate fake metrics
            for _ in range(self.num_samples // (batchsize * trainer.world_size)):
                labels = next(label_iter).to(device)
                noise = training_utils.generate_noise(batchsize, noise_channels).to(device)
                with torch.no_grad():
                    img = generator(noise, init_points, labels)
                img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                self.metrics.update(img, real=False)

            metrics = self.metrics.compute()
            pl_module.log_dict(metrics, on_epoch=True, on_step=False)

    def calculate_real_metrics(self, dataloader, device):
        @rank_zero_only
        def save_metric(metric, path):
            metric.sync()
            state = metric.state_dict()
            torch.save(state, path)
            logger.info(f"Saved {metric} metric to {path}")

        @rank_zero_only
        def calculate_metric():
            logger.info("Calculating real features for metrics...")
            for img, _ in dataloader:
                img = img.to(device)
                img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                self.metrics.update(img, real=True)

        # load state if available
        metrics_to_load = []
        if self.cache_path.exists() and not self.recalculate:
            for name in self.metrics.keys():
                path = f"{self.cache_path}/{name}.pt"
                if Path(path).exists():
                    metrics_to_load.append(name)

        if len(metrics_to_load) != len(self.metrics):
            calculate_metric()

        self.cache_path.mkdir(parents=True, exist_ok=True)
        for name, metric in self.metrics.items():
            path = f"{self.cache_path}/{name}.pt"
            if name in metrics_to_load:
                metric.load_state_dict(torch.load(path, map_location=device))
                logger.info(f"Loaded {name} metric")
            else:
                save_metric(metric, path)
