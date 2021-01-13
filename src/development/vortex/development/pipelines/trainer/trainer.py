import warnings
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers

from pathlib import Path
from copy import deepcopy
from easydict import EasyDict

from pytorch_lightning.trainer.connectors.logger_connector import LoggerConnector
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from .checkpoint import CheckpointConnector
from .patches import (
    patch_checkpoint_filepath_name,
    patch_checkpoint_backward_monitor,
    patch_configure_logger,
    patch_trainer_on_load_checkpoint,
    patch_trainer_on_save_checkpoint
)
from vortex.development.utils.factory import create_model, create_dataloader
from vortex.development.networks.models import ModelBase


class TrainingPipeline:
    def __init__(self, config):
        super().__init__()

        ## TODO: validate config
        self.config = config

        self.model = self.create_model(self.config)

        ## TODO: fix progress bar

        ## TODO: log lr with 'LearningRateMonitor'

        self.experiment_dir = str(Path('.').joinpath("experiments", config['experiment_name']))
        self.trainer = self.create_trainer(self.experiment_dir, self.config, self.model)

        self.train_dataloader, self.val_dataloader = self.create_dataloaders(self.config, self.model)
        self._copy_data_to_model(self.train_dataloader, self.config, self.model)


    def run(self):
        self.trainer.fit(self.model, self.train_dataloader, self.val_dataloader)


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
    def create_model_checkpoints(config: dict, model: ModelBase):
        fname_prefix = config['experiment_name']

        ## patches for model checkpoint
        ModelCheckpoint.FILE_EXTENSION = ".pth"
        ModelCheckpoint._add_backward_monitor_support = patch_checkpoint_backward_monitor
        ModelCheckpoint._get_metric_interpolated_filepath_name = patch_checkpoint_filepath_name

        callbacks = [
            ## default checkpoint callback: save last epoch
            ModelCheckpoint(
                filename=fname_prefix+"-last", monitor=None,
                save_top_k=None, mode="min"
            )
        ]

        if 'save_epoch' in config['trainer'] and config['trainer']['save_epoch'] is not None:
            save_epoch = int(config['trainer']['save_epoch'])
            if save_epoch < 1:
                raise RuntimeError("Invalid value in 'config.trainer.save_epoch' of {}, "
                    "expected value of integer higher than 0 (> 0).".format(save_epoch))
            epoch_ckpt_callback = ModelCheckpoint(
                filename=fname_prefix+"-{epoch}", monitor=None,
                save_top_k=None, mode="min",
                period=save_epoch
            )
            epoch_ckpt_callback.save_epoch = True   ## to differentiate with last epoch ckpt
            callbacks.append(epoch_ckpt_callback)

        available_metrics = model.available_metrics
        if isinstance(available_metrics, list):
            warnings.warn("'model.available_metrics()' returns list, so it doesn't describe optimization "
                "strategy to use ('min' or 'max').\nWill infer from metrics name with metrics that contains "
                "'loss' in the name will use 'min' strategy otherwise use 'max'.\nMake sure it is correctly "
                "configured.")
            available_metrics = {m: "min" if "loss" in m else "max" for m in available_metrics}

        if 'save_best_metrics' in config['trainer'] and config['trainer']['save_best_metrics']:
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
                    mode=available_metrics[m]
                ))
        return callbacks


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

        logger_cfg = None
        if "logging" in config:
            logger_cfg = config["logging"]
        if logger_cfg is None or no_log:
            return False

        if not isinstance(logger_cfg, (dict, EasyDict)):
            raise RuntimeError("Unknown data type of 'config.logging' field, expected to have "
                "'dict' type, got {}".format(type(logger_cfg)))
        if "module" not in logger_cfg or "args" not in logger_cfg:
            raise RuntimeError("logger config is incomplete, 'config.logging' is expected to have "
                "'module' and 'args' attribute, got {}".format(list(logger_cfg)))

        logger_module = None
        if logger_cfg["module"] in logger_map:
            logger_module = logger_map[logger_cfg["module"]]
        if hasattr(pl_loggers, logger_cfg["module"]):
            logger_module = getattr(pl_loggers, logger_cfg["module"])
        else:
            raise RuntimeError("Unknown logger module name of '{}', available logger: {}"
                .format(logger_cfg["module"], list(logger_map)))
        return logger_module(save_dir=experiment_dir, **logger_cfg["args"])

    @staticmethod
    def create_lr_monitor(config):
        return LearningRateMonitor()

    @staticmethod
    def create_trainer(experiment_dir, config, model, no_log=False) -> pl.Trainer:
        trainer_args = dict()
        trainer_args.update(TrainingPipeline._decide_device_to_use(config))
        trainer_args.update(TrainingPipeline._handle_validation_interval(config))

        if 'args' in config.trainer and config.trainer.args is not None:
            trainer_args.update(config.trainer.args)

        callbacks = TrainingPipeline.create_model_checkpoints(config, model)
        callbacks.append(TrainingPipeline.create_lr_monitor(config))
        loggers = TrainingPipeline.create_loggers(experiment_dir, config, no_log)

        TrainingPipeline._patch_trainer_components()

        trainer = pl.Trainer(
            max_epochs=config['trainer']['epoch'],
            default_root_dir=experiment_dir,
            deterministic=True, benchmark=True,
            weights_summary=None,
            callbacks=callbacks,
            logger=loggers,
            **trainer_args
        )

        TrainingPipeline._patch_trainer_object(trainer)

        return trainer

    @staticmethod
    def _patch_trainer_components():
        ## patch for logger path (exclude 'lightning_logs' when logger not set)
        LoggerConnector.configure_logger = patch_configure_logger

        ## patch for multiple model checkpoint support
        pl.Trainer.on_save_checkpoint = patch_trainer_on_save_checkpoint
        pl.Trainer.on_load_checkpoint = patch_trainer_on_load_checkpoint

    @staticmethod
    def _patch_trainer_object(trainer: pl.Trainer):
        ## patch for additional checkpoint data
        trainer.checkpoint_connector = CheckpointConnector(trainer)
        return trainer

    @staticmethod
    def run_sanity_check(model, train_dataloader, val_dataloader=None, **trainer_kwargs):
        trainer_sanity = pl.Trainer(
            gpus=None,
            logger=False, checkpoint_callback=False,
            limit_train_batches=2, limit_val_batches=2,
            progress_bar_refresh_rate=0, max_epochs=1,
            num_sanity_val_steps=0, weights_summary=None,
            **trainer_kwargs
        )
        trainer_sanity.fit(deepcopy(model), train_dataloader, val_dataloader)
        return trainer_sanity

    @staticmethod
    def _decide_device_to_use(config):
        gpus, auto_select_gpus = None, False
        device = None
        if 'device' in config:
            device = config['device']

        if device is not None and ('cuda' in device or 'gpu' in device):
            len_device = len(device.split(':'))
            if len_device == 1:
                gpus, auto_select_gpus = 1, True
            elif len_device == 2:
                gpus = device.split(':')[-1]
            else:
                raise RuntimeError("Unknown 'device' argument of {}".format(device))

        return {
            'gpus': gpus,
            'auto_select_gpus': auto_select_gpus
        }

    @staticmethod
    def _copy_data_to_model(dataloader, config, model):
        class_names = None
        if hasattr(dataloader.dataset, "class_names"):
            class_names = dataloader.dataset.class_names
        elif hasattr(dataloader.dataset, "classes"):
            class_names = dataloader.dataset.classes
        model.class_names = class_names

        model.config = deepcopy(config)

    @staticmethod
    def _handle_validation_interval(config):
        val_epoch = 1
        if 'validator' in config and 'val_epoch' in config['validator']:
            try:
                val_epoch = int(config['validator']['val_epoch'])
            except ValueError:
                raise RuntimeError("Unknown value in 'config.validator.val_epoch' of {}"
                    .format(config['validator']['val_epoch']))
        elif 'trainer' in config and 'validate_interval' in config['trainer']:
            try:
                val_epoch = int(config['trainer']['validate_interval'])
            except ValueError:
                raise RuntimeError(f"Unknown value in 'config.trainer.validate_interval' "
                    "of {}".format(config['trainer']['validate_interval']))

        return dict(check_val_every_n_epoch=val_epoch)
