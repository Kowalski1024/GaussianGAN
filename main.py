from pathlib import Path

import hydra
from omegaconf import OmegaConf, DictConfig
import torch
from pytorch_lightning.strategies import DDPStrategy

from src.network.generator import ImageGenerator

from src.utils.training import get_dataset
from conf.main_config import MainConfig
from pytorch_lightning.trainer import Trainer, seed_everything
from src.utils.pylogger import RankedLogger
from src.utils.instantiators import instantiate_loggers
from src.loss import GANLoss
from src.network.networks_stylegan2 import Discriminator

logger = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(config_path="conf", config_name="main_config", version_base="1.3")
def main(cfg: MainConfig) -> None:
    """Main function for the pruning entry point

    Args:
        cfg (MainConfig): Hydra config object with all the settings. (Located in config/main_config.py)
    """
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")
    hydra_output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    logger.info(f"Hydra output directory: {hydra_output_dir}")

    seed_everything(seed=cfg.seed)

    loggers = instantiate_loggers(cfg.logger)

    generator = ImageGenerator(
        generator_config=cfg.generator,
        image_size=cfg.dataset.image_size,
        background=cfg.dataset.background,
    )
    discriminator = Discriminator()
    dataset = get_dataset(cfg.dataset)
    print(generator)

    model = GANLoss(
        **cfg.training.loss,
        generator=generator,
        discriminator=discriminator,
        dataset=dataset,
        main_config=cfg,
    )
    trainer = Trainer(
        max_epochs=10,
        limit_train_batches=16,
        log_every_n_steps=4,
        logger=loggers,
        strategy=DDPStrategy(find_unused_parameters=True),
        enable_progress_bar=False,
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
