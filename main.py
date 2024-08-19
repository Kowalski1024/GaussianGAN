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
from src.utils.callbacks import EMACallback, ImgCheckpointCallback
from src.utils.instantiators import instantiate_loggers
from src.utils.pylogger import RankedLogger
from src.utils.training import get_dataset

logger = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(config_path="conf", config_name="main_config", version_base="1.3")
def main(cfg: MainConfig) -> None:
    """Main function for the pruning entry point

    Args:
        cfg (MainConfig): Hydra config object with all the settings. (Located in config/main_config.py)
    """
    torch.set_float32_matmul_precision("high")
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")
    hydra_output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    logger.info(f"Hydra output directory: {hydra_output_dir}")

    seed_everything(seed=cfg.seed)

    loggers = instantiate_loggers(cfg.logger)

    dataset = get_dataset(cfg.dataset)

    generator = ImageGenerator(
        generator_config=cfg.generator,
        image_size=dataset.image_size,
        background=dataset.background,
    )
    discriminator = Discriminator()
    print(generator)

    # callbacks
    ema_callback = EMACallback()
    img_checkpoint_callback = ImgCheckpointCallback(
        dataset,
        hydra_output_dir / "images",
        interval=cfg.training.image_save_interval,
    )

    model = GANLoss(
        **cfg.training.loss,
        generator=generator,
        discriminator=discriminator,
        dataset=dataset,
        main_config=cfg,
    )

    trainer = Trainer(
        max_epochs=1000,
        limit_train_batches=64,
        log_every_n_steps=16,
        logger=loggers,
        strategy=DDPStrategy(find_unused_parameters=True),
        enable_progress_bar=cfg.enable_progress_bar,
        callbacks=[ema_callback, img_checkpoint_callback],
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
