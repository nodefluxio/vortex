import os
import pytorch_lightning as pl

from typing import Dict, Any, Iterable
from copy import deepcopy

from pytorch_lightning.trainer.connectors.logger_connector import LoggerConnector
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger


def patch_configure_logger(self: LoggerConnector, logger):
    if logger is True:
        version = os.environ.get('PL_EXP_VERSION', self.trainer.slurm_job_id)

        # default logger
        self.trainer.logger = TensorBoardLogger(
            save_dir=self.trainer.default_root_dir, version=version, name=None
        )
    elif logger is False:
        self.trainer.logger = None
    else:
        if isinstance(logger, Iterable):
            self.trainer.logger = LoggerCollection(logger)
        else:
            self.trainer.logger = logger

def patch_trainer_on_save_checkpoint(self: pl.Trainer):
    """patch for multiple `ModelCheckpoint` object on save.
    """
    callback_states = {}
    for callback in self.callbacks:
        if isinstance(callback, ModelCheckpoint):
            monitor = callback.monitor if callback.monitor else "last"
            callback_class = "checkpoint_" + monitor
        else:
            callback_class = type(callback)
        state = callback.on_save_checkpoint(self, self.get_model())
        if state:
            callback_states[callback_class] = state
    return callback_states

def patch_trainer_on_load_checkpoint(self: pl.Trainer, checkpoint: dict):
    """patch for multiple `ModelCheckpoint` object on load.
    """
    callback_states = checkpoint.get('callbacks')
    for callback in self.callbacks:
        if isinstance(callback, ModelCheckpoint):
            monitor = callback.monitor if callback.monitor else "last"
            state = callback_states.get("checkpoint_" + monitor)
        else:
            state = callback_states.get(type(callback))
        if state:
            state = deepcopy(state)
            callback.on_load_checkpoint(state)


def patch_checkpoint_filepath_name(self: ModelCheckpoint, ckpt_name_metrics: Dict[str, Any], epoch: int, step: int):
    """disable topk save (remove '-v0' filename) from model checkpoint
    """
    return self.format_checkpoint_name(epoch, step, ckpt_name_metrics)

def patch_checkpoint_backward_monitor(self, trainer):
    """
    this disables backward compatibility in checkpoint callback,
    which caused when save last epoch (ModelCheckpoint(monitor=None, save_top_k=None))
    is used and there is 'val_loss' available in metrics, it will be
    forced to monitor 'val_loss' instead, which we don't want.
    """
    pass
