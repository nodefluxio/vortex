import warnings
import torch.nn as nn
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers

from pathlib import Path
from copy import deepcopy
from easydict import EasyDict

from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector
from pytorch_lightning.trainer.connectors.logger_connector import LoggerConnector
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from .checkpoint import CheckpointConnector
from .patches import (
    patch_checkpoint_filepath_name,
    patch_checkpoint_backward_monitor,
    patch_configure_logger,
    patch_trainer_on_load_checkpoint,
    patch_trainer_on_save_checkpoint
)
from vortex.development.core import create_model, create_dataloader
from vortex.development.networks.models import ModelBase


class TrainingPipeline:
    def __init__(self, config):
        super().__init__()

        ## TODO: validate config
        self.config = config

        ## TODO: build model
        self.model = self.create_model(self.config)
        self.model.config = self.config

        experiment_dir = str(Path('.').joinpath("experiments", config['experiment_name']))
        self.trainer = self.create_trainer(experiment_dir)

    def run(self):
        train_loader, val_loader = self.create_dataloaders(self.config, self.model)

        self.trainer.fit(self.model, train_loader, val_loader)


    @staticmethod
    def create_model(config) -> ModelBase:
        model = create_model(config.model)
        if isinstance(model, pl.LightningModule):
            raise RuntimeError("model '{}' is not a 'LightningModule' subclass. "
                "Please update it to inherit 'LightningModule', see more in "
                "https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html"
                .format(config.model.name))
        return model

    @staticmethod
    def create_dataloaders(config, model):
        assert model is not None, "Model is not initiated"

        train_dataloader, val_dataloader = None, None
        if 'train' in config.dataset:
            train_dataloader = create_dataloader(
                config.dataloader, config.dataset,
                preprocess_config=config.model.preprocess_config,
                collate_fn=model.collate_fn,
                stage='train'
            )
        else:
            raise RuntimeError("Train dataset config (config.dataset.train) is not found, "
                "Please specify it properly")

        if 'val' in config.dataset:
            config.dataset.eval = config.dataset.val
        if 'eval' in config.dataset:
            val_dataloader = create_dataloader(
                config.dataloader, config.dataset,
                preprocess_config=config.model.preprocess_config,
                collate_fn=model.collate_fn,
                stage='validate'
            )

        return train_dataloader, val_dataloader


    @staticmethod
    def create_model_checkpoints(experiment_dir, config, model):
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
                dirpath=experiment_dir
            )
        ]

        available_metrics = model.available_metrics
        if isinstance(available_metrics, list):
            warnings.warn("'model.available_metrics()' returns list, so it doesn't describe optimization "
                "strategy to use ('min' or 'max').\nWill infer from metrics name with metrics that contains "
                "'loss' in the name will use 'min' strategy otherwise use 'max'.\nMake sure it is correctly "
                "configured.")
            available_metrics = {m: "min" if "loss" in m else "max" for m in available_metrics}

        save_best_metrics = config['trainer']['save_best_metrics']
        if isinstance(save_best_metrics, str):
            save_best_metrics = [save_best_metrics]

        for m in save_best_metrics:
            if not m in available_metrics:
                raise RuntimeError("metric '{}' is not available to track for 'save_best_metrics' "
                    "argument, available metrics: {}".format(m, list(available_metrics.keys())))
            callbacks.append(ModelCheckpoint(
                filename=fname_prefix + "-best_" + m,
                monitor=m,
                mode=available_metrics[m],
                dirpath=experiment_dir
            ))


    @staticmethod
    def create_loggers(experiment_dir, config, no_log=False):
        logger_map = {
            'comet_ml': pl_loggers.CometLogger,
            'ml_flow': pl_loggers.MLFlowLogger,
            'neptune': pl_loggers.NeptuneLogger,
            'tensorboard': pl_loggers.TensorBoardLogger,
            'test_tube': pl_loggers.TestTubeLogger,
            'wandb': pl_loggers.WandbLogger,
            'csv_logger': pl_loggers.CSVLogger
        }

        logger_cfg = config["logging"]
        if logger_cfg is None or no_log:
            return False

        if not isinstance(logger_cfg, (dict, EasyDict)):
            raise RuntimeError("Unknown data type of 'config.logging' field, expected to have "
                "'dict' type, got {}".format(type(logger_cfg)))
        if "module" not in logger_cfg or "args" not in logger_cfg:
            raise RuntimeError("logger config is incomplete, 'config.logging' is expected to have "
                "'module' and 'args' attribute, got {}".format(list(logger_cfg)))

        loggers = None
        logger_module = None
        if logger_cfg["module"] in logger_map:
            logger_module = logger_map[logger_cfg["module"]]
        if hasattr(logger_cfg["module"]):
            logger_module = logger_cfg["module"]
            loggers = logger_module(**logger_cfg["args"])
        else:
            raise RuntimeError("Unknown logger module name of '{}', available logger: {}"
                .format(logger_cfg["module"], list(logger_map)))
        return loggers

    def create_trainer(self, experiment_dir, config=None, model=None) -> pl.Trainer:
        if config:
            self.config = config
        if model:
            self.model = model
        self.experiment_dir = experiment_dir

        trainer_args = dict()
        trainer_args.update(self._decide_device_to_use())

        if 'args' in self.config.trainer and self.config.trainer.args is not None:
            trainer_args.update(self.config.trainer.args)

        callbacks = self.create_model_checkpoints(self.experiment_dir, self.config, self.model)
        loggers = self.create_loggers(self.experiment_dir, self.config)

        ## patch for logger path (exclude 'lightning_logs' when logger not set)
        LoggerConnector.configure_logger = patch_configure_logger
        pl.Trainer.on_save_checkpoint = patch_trainer_on_save_checkpoint
        pl.Trainer.on_load_checkpoint = patch_trainer_on_load_checkpoint

        trainer = pl.Trainer(
            max_epochs=self.config['trainer']['epoch'],
            logger=loggers,
            default_root_dir=self.experiment_dir,
            deterministic=True, 
            benchmark=True,
            weights_summary=None,
            move_metrics_to_cpu=True,
            callbacks=callbacks,
            logger=loggers,
            **trainer_args
        )
        ## patch for additional checkpoint data
        trainer.checkpoint_connector = CheckpointConnector(trainer)

        return trainer

    def _decide_device_to_use(self, device=None):
        gpus, auto_select_gpus = None, False
        if device is None and 'device' in self.config:
            device = self.config.device

        if device is not None and ('cuda' in device or 'gpu' in device):
            len_device = len(device.split(':'))
            if len_device == 1:
                gpus, auto_select_gpus = 1, True
            elif len_device == 2:
                gpus = device.split(':')[-1]
            else:
                raise RuntimeError("Unknown 'device' argument of {}".format(device))
        kwargs = {
            'gpus': gpus,
            'auto_select_gpus': auto_select_gpus
        }
        return kwargs


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
