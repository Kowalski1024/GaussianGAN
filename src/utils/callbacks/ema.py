from typing import TYPE_CHECKING

from ema_pytorch import EMA
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from src.utils.pylogger import RankedLogger

if TYPE_CHECKING:
    from src.loss import GANLoss

logger = RankedLogger(__name__, rank_zero_only=True)


class EMACallback(Callback):
    def __init__(self, decay=0.9999, use_ema_weights: bool = True):
        self.decay = decay
        self.ema = None
        self.use_ema_weights = use_ema_weights

    def on_fit_start(self, trainer: pl.Trainer, pl_module: "GANLoss"):
        self.ema = EMA(pl_module.generator, beta=self.decay, allow_different_devices=True)

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: "GANLoss", outputs, batch, batch_idx
    ):
        "Update the stored parameters using a moving average"
        # Update currently maintained parameters.
        self.ema.update()

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: "GANLoss"):
        "do validation using the stored parameters"
        logger.info("Replacing model weights with EMA version for validation.")
        # save original parameters before replacing with EMA version
        self.store(pl_module.generator.parameters())

        # update the LightningModule with the EMA weights
        # ~ Copy EMA parameters to LightningModule
        self.copy_to(self.ema.parameters(), pl_module.generator.parameters())

    def on_validation_end(self, trainer: pl.Trainer, pl_module: "GANLoss"):
        "Restore original parameters to resume training later"
        self.restore(pl_module.generator.parameters())

    def on_train_end(self, trainer: pl.Trainer, pl_module: "GANLoss"):
        # update the LightningModule with the EMA weights
        if self.use_ema_weights:
            self.copy_to(self.ema.parameters(), pl_module.generator.parameters())
            msg = "Model weights replaced with the EMA version."
            logger.info(msg)

    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: "GANLoss", checkpoint):
        if self.ema is not None:
            return {"state_dict_ema": self.ema.state_dict()}

    def on_load_checkpoint(self, callback_state):
        if self.ema is not None:
            self.ema.load_state_dict(callback_state["state_dict_ema"])

    def store(self, parameters):
        "Save the current parameters for restoring later."
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def copy_to(self, shadow_parameters, parameters):
        "Copy current parameters into given collection of parameters."
        for s_param, param in zip(shadow_parameters, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)
