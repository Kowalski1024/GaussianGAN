import logging
from pathlib import Path

import hydra
from omegaconf import OmegaConf, DictConfig
import torch

from conf.main_config import MainConfig
from src.train import LSGAN
from pytorch_lightning.trainer import Trainer
from src.utils.pylogger import RankedLogger
from src.utils.instantiators import instantiate_loggers

logger = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(config_path="conf", config_name="main_config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main function for the pruning entry point

    Args:
        cfg (MainConfig): Hydra config object with all the settings. (Located in config/main_config.py)
    """
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")
    hydra_output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    logger.info(f"Hydra output directory: {hydra_output_dir}")

    loggers = instantiate_loggers(cfg.logger)

    model = LSGAN(cfg)
    trainer = Trainer(max_epochs=100, limit_train_batches=10, log_every_n_steps=4, logger=loggers)
    trainer.fit(model)


if __name__ == "__main__":
    main()
