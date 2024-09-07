from pathlib import Path

import hydra
from omegaconf import OmegaConf
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.trainer import Trainer, seed_everything
import torch

from conf.main_config import MainConfig
from src.loss import GANLoss
from src.network.generator import ImageGenerator
from src.network.networks_stylegan2 import Discriminator
import src.utils.callbacks as callbacks
from src.utils.instantiators import instantiate_loggers
from src.utils.pylogger import RankedLogger
from src.utils.training import download_inception_model, get_dataset

logger = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(config_path="conf", config_name="main_config", version_base="1.3")
def main(cfg: MainConfig) -> None:
    """Main function for the pruning entry point

    Args:
        cfg (MainConfig): Hydra config object with all the settings. (Located in config/main_config.py)
    """
    torch.set_float32_matmul_precision("high")
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")
    hydra_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger.info(f"Hydra output directory: {hydra_output_dir}")

    seed_everything(seed=cfg.seed)
    download_inception_model()

    loggers = instantiate_loggers(cfg.logger)
    train_dataset = get_dataset(cfg.dataset, subset_type=("train",))
    test_dataset = get_dataset(cfg.dataset, subset_type=("test",))

    generator = ImageGenerator(
        generator_config=cfg.generator,
        image_size=train_dataset.image_size,
        background=train_dataset.background,
    )
    discriminator = Discriminator()
    print(generator)

    # callbacks
    ema_callback = callbacks.EMACallback()
    img_checkpoint_callback = callbacks.ImgCheckpointCallback(
        train_dataset,
        hydra_output_dir / "images",
        interval=cfg.training.save_img_every_n_epoch,
    )
    metrics_callback = callbacks.MetricsCallback(
        selected_metrics=["fid"],
        cached_features_path=f"cache/{train_dataset.name}",
        interval=cfg.training.metric_every_n_epoch,
        num_samples=50000,
    )

    model = GANLoss(
        **cfg.training.loss,
        generator=generator,
        discriminator=discriminator,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        main_config=cfg,
    )

    trainer = Trainer(
        max_epochs=cfg.training.max_epochs,
        limit_train_batches=cfg.training.limit_train_batches,
        log_every_n_steps=cfg.training.log_every_n_steps,
        check_val_every_n_epoch=10,
        limit_val_batches=1,
        logger=loggers,
        strategy=DDPStrategy(find_unused_parameters=True),
        enable_progress_bar=cfg.enable_progress_bar,
        callbacks=[ema_callback, img_checkpoint_callback, metrics_callback],
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
