import pytorch_lightning as pl

from pathlib import Path
from copy import deepcopy

from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector
from pytorch_lightning.trainer.connectors.logger_connector import LoggerConnector
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger


from .checkpoint import CheckpointConnector
from .patches import (
    patch_checkpoint_filepath_name,
    patch_checkpoint_backward_monitor,
    patch_configure_logger,
    patch_trainer_on_load_checkpoint,
    patch_trainer_on_save_checkpoint
)


class TrainingPipeline:
    def __init__(self, config):
        super().__init__()

        ## TODO: build model
        self.model = None

        ## TODO: validate config
        self.config = config

        experiment_dir = str(Path('.').joinpath("experiments", config['experiment_name']))
        self.trainer = self.create_trainer(experiment_dir)

        ## create dataset
        self.datamodule = None


    def run(self):
        pass

    def create_trainer(self, experiment_dir, config=None) -> pl.Trainer:
        config = config if config else self.config
        self.experiment_dir = experiment_dir
        fname_prefix = config['experiment_name']

        ## patches for model checkpoint
        ModelCheckpoint.FILE_EXTENSION = ".pth"
        ModelCheckpoint._add_backward_monitor_support = patch_checkpoint_backward_monitor
        ModelCheckpoint._get_metric_interpolated_filepath_name = patch_checkpoint_filepath_name

        callbacks = [
            ## default checkpoint callback: save last epoch
            ModelCheckpoint(
                filename=fname_prefix+"-last", monitor=None,
                save_top_k=None, mode="min",
            )
        ]

        available_metrics = self.model.available_metrics
        if isinstance(available_metrics, list):
            available_metrics = {m: "min" if "loss" in m else "max" for m in available_metrics}

        save_best_metrics = config['trainer']['save_best_metrics']
        if isinstance(save_best_metrics, str):
            save_best_metrics = [save_best_metrics]

        for m in save_best_metrics:
            if not m in available_metrics:
                raise RuntimeError("metric '{}' is not available to track for 'save_best_metrics' "
                    "argument, available metrics: {}".format(m, list(available_metrics.keys())))
            callbacks.append(pl_callbacks.ModelCheckpoint(
                filename=fname_prefix + "-best_" + m,
                monitor=m,
                mode=available_metrics[m]
            ))

        ## TODO: create loggers
        loggers = True

        ## patch for logger path (exclude 'lightning_logs' when logger not set)
        LoggerConnector.configure_logger = patch_configure_logger
        pl.Trainer.on_save_checkpoint = patch_trainer_on_save_checkpoint
        pl.Trainer.on_load_checkpoint = patch_trainer_on_load_checkpoint

        trainer = pl.Trainer(
            gpus=1,
            max_epochs=config['trainer']['epoch'],
            logger=loggers,
            default_root_dir=self.experiment_dir,
            deterministic=True, 
            benchmark=True,
            weights_summary=None,
            move_metrics_to_cpu=True,
            callbacks=callbacks,
        )
        ## patch for additional checkpoint data
        trainer.checkpoint_connector = CheckpointConnector(trainer)

        return trainer

    def run_sanity_check(self, **trainer_kwargs):
        trainer_sanity = pl.Trainer(
            gpus=1,
            logger=False, checkpoint_callback=False,
            limit_train_batches=2, limit_val_batches=2,
            progress_bar_refresh_rate=0, max_epochs=1,
            num_sanity_val_steps=0, weights_summary=None,
            **trainer_kwargs
        )
        trainer_sanity.fit(deepcopy(self.model), self.datamodule)
        return trainer_sanity
