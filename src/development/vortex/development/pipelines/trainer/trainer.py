import warnings
import logging
import shutil
import yaml
import torch
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers

from typing import Union
from pathlib import Path
from copy import deepcopy
from easydict import EasyDict
from packaging.version import parse as parse_version

from pytorch_lightning.trainer.connectors.logger_connector import LoggerConnector
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from .checkpoint import CheckpointConnector, AlwaysSaveCheckpointCallback
from .progress import VortexProgressBar
from .patches import (
    patch_checkpoint_filepath_name,
    patch_checkpoint_backward_monitor,
    patch_configure_logger,
    patch_trainer_on_load_checkpoint,
    patch_trainer_on_save_checkpoint
)
from vortex.development.utils.common import easydict_to_dict
from vortex.development.pipelines.base_pipeline import BasePipeline
from vortex.development.utils.parser import load_config, check_config
from vortex.development.utils.factory import create_model, create_dataloader
from vortex.development.networks.models import ModelBase

LOGGER = logging.getLogger(__name__)


class TrainingPipeline(BasePipeline):
    def __init__(self, config: Union[str, Path, dict], hypopt=False, resume=False, no_log=False):
        super().__init__()

        self.hypopt = hypopt
        self.no_log = no_log
        self.resume = resume

        self.config = self.load_config(config)
        self._check_experiment_config(self.config)

        resume_checkpoint_path, state_dict = self._handle_resume_checkpoint(self.config, resume)
        self.model = self.create_model(self.config, state_dict)

        ## TODO: change default vortex root with environment variable

        self.experiment_dir = self._format_experiment_dir(self.config)
        if not no_log:
            self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.trainer = self.create_trainer(
            str(self.experiment_dir), self.config, self.model,
            hypopt=hypopt, no_log=no_log,
            resume_checkpoint_path=resume_checkpoint_path
        )

        self.train_dataloader, self.val_dataloader = self.create_dataloaders(self.config, self.model)
        self._copy_data_to_model(self.train_dataloader, self.config, self.model)

        if self.trainer.logger is None:
            self.experiment_version = "version_0"
            self.run_directory = self.experiment_dir.joinpath(self.experiment_version)
        else:
            self.experiment_version = (
                self.trainer.logger.version
                if isinstance(self.trainer.logger.version, str)
                else f"version_{self.trainer.logger.version}"
            )
            self.run_directory = self.experiment_dir.joinpath(self.experiment_version)

        if not hypopt:
            if not no_log:  ## TODO: should we dump config when not log?
                self._dump_config(self.config, self.run_directory)
            print("\nExperiment directory:", self.run_directory)

    def run(self):
        self.trainer.fit(self.model, self.train_dataloader, self.val_dataloader)
        self._copy_final_checkpoint(self.trainer, self.config, self.experiment_dir, self.hypopt)


    @staticmethod
    def create_model(config, state_dict=None) -> ModelBase:
        if not isinstance(config, EasyDict):
            config = EasyDict(config)
        model = create_model(config.model, state_dict=state_dict)
        if not isinstance(model, pl.LightningModule):
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
                preprocess_config=config.model.preprocess_args,
                collate_fn=model.collate_fn,
                stage='train'
            )
        else:
            raise RuntimeError("Train dataset config ('config.dataset.train') is not found, "
                "please specify it properly")

        if 'eval' in config.dataset and config.dataset.eval is not None:
            val_dataloader = create_dataloader(
                config.dataloader, config.dataset,
                preprocess_config=config.model.preprocess_args,
                collate_fn=model.collate_fn,
                stage='validate',
                force_normalize=True
            )

        return train_dataloader, val_dataloader


    @staticmethod
    def create_model_checkpoints(config: dict, model: ModelBase):
        fname_prefix = config['experiment_name']

        TrainingPipeline._patch_model_checkpoint()
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
                save_top_k=-1, mode="min",
                period=save_epoch
            )
            epoch_ckpt_callback.save_epoch = True   ## to differentiate with last epoch ckpt
            callbacks.append(epoch_ckpt_callback)

        available_metrics = model.available_metrics
        if isinstance(available_metrics, list):
            LOGGER.warning("'model.available_metrics()' returns list, so it doesn't describe optimization "
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
                    filename=fname_prefix + "-best_{" + m + ":.2f}",
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
        if no_log:
            return False
        elif "logging" in config:
            logger_cfg = config["logging"]
        elif "logger" in config["trainer"]:
            logger_cfg = config["trainer"]["logger"]

        if logger_cfg is None:
            ## use default (tensorboard) logger
            return True
        if isinstance(logger_cfg, bool):
            return logger_cfg

        if not isinstance(logger_cfg, (dict, EasyDict)):
            raise TypeError("Unknown data type of 'config.logging' field, expected to have "
                "'dict' type, got {}".format(type(logger_cfg)))
        if "module" not in logger_cfg:
            raise RuntimeError("logger config is incomplete, 'config.logging' is expected to have "
                "'module' and 'args' attribute, got {}".format(list(logger_cfg)))

        logger_module = logger_cfg["module"]
        if logger_module in logger_map:
            logger_module = logger_map[logger_module]
        elif hasattr(pl_loggers, logger_module):
            logger_module = getattr(pl_loggers, logger_module)
        else:
            raise RuntimeError("Unknown logger module name of '{}', available logger: {}"
                .format(logger_module, list(logger_map)))
        logger_args = {}
        if "args" in logger_cfg:
            logger_args = logger_cfg["args"]
        logger = logger_module(save_dir=experiment_dir, **logger_args)
        return logger

    @staticmethod
    def create_lr_monitor(config):
        return LearningRateMonitor()

    @staticmethod
    def create_progeress_bar(config):
        return VortexProgressBar("TRAIN")

    @staticmethod
    def create_trainer(
        experiment_dir, config, model, no_log=False,
        hypopt=False, resume_checkpoint_path=False
    ) -> pl.Trainer:

        trainer_args = dict()
        trainer_args.update(TrainingPipeline._trainer_args_device(config))
        trainer_args.update(TrainingPipeline._trainer_args_validation_interval(config))
        trainer_args.update(TrainingPipeline._trainer_args_set_seed(config))
        trainer_args.update(TrainingPipeline._trainer_args_accumulate_grad(config))

        if 'args' in config.trainer and config.trainer.args is not None:
            if isinstance(config.trainer.args, dict):
                trainer_args.update(config.trainer.args)
            else:
                raise TypeError("Unknown type for 'config.trainer.args', expected to have "
                    "dict type, got '{}' with value of {}".format(type(config.trainer.args), 
                    config.trainer.args))

        loggers = False
        callbacks = []
        checkpoint_callback = False
        if not hypopt:
            checkpoint_callback = True
            callbacks = TrainingPipeline.create_model_checkpoints(config, model)
            callbacks.append(TrainingPipeline.create_lr_monitor(config))
            callbacks.append(TrainingPipeline.create_progeress_bar(config))
            callbacks.append(AlwaysSaveCheckpointCallback())
            loggers = TrainingPipeline.create_loggers(experiment_dir, config, no_log)

        TrainingPipeline._patch_trainer_components()
        kwargs = dict(
            max_epochs=config['trainer']['epoch'],
            default_root_dir=experiment_dir,
            benchmark=False,
            weights_summary=None,
            callbacks=callbacks,
            checkpoint_callback=checkpoint_callback,
            logger=loggers,
            resume_from_checkpoint=resume_checkpoint_path
        )
        kwargs.update(trainer_args)
        trainer = pl.Trainer(**kwargs)
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
    def _patch_model_checkpoint():
        ## patches for model checkpoint
        ModelCheckpoint.FILE_EXTENSION = ".pth"
        ModelCheckpoint._add_backward_monitor_support = patch_checkpoint_backward_monitor
        if parse_version(pl.__version__) < parse_version('1.1.2'):
            ## this patch fixed in 1.1.2
            ModelCheckpoint._get_metric_interpolated_filepath_name = patch_checkpoint_filepath_name

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
    def _trainer_args_device(config):
        ## TODO:
        ## - support multi gpus select ('cuda:0,1')
        ## - trainer only device (or gpu) config -> config.trainer.gpus
        gpus, auto_select_gpus = None, False
        device = None
        if 'device' in config:
            device = config['device']

        if device is None or device == 'cpu':
            gpus, auto_select_gpus = None, False
        elif device is not None and ('cuda' in device or 'gpu' in device):
            len_device = len(device.split(':'))
            if len_device == 1:
                gpus, auto_select_gpus = 1, True
            elif len_device == 2:
                gpus = device.split(':')[-1]
            else:
                raise RuntimeError("Unknown 'device' argument of {}".format(device))

        return dict(gpus=gpus, auto_select_gpus=auto_select_gpus)

    @staticmethod
    def _trainer_args_validation_interval(config):
        val_epoch = 1
        set_in_validator = False
        if 'validator' in config and 'val_epoch' in config['validator']:
            set_in_validator = True
            try:
                val_epoch = int(config['validator']['val_epoch'])
            except ValueError:
                raise ValueError("Unknown value in 'config.validator.val_epoch' of {}"
                    .format(config['validator']['val_epoch']))
        elif 'trainer' in config and 'validation_interval' in config['trainer']:
            try:
                val_epoch = int(config['trainer']['validation_interval'])
            except ValueError:
                raise ValueError(f"Unknown value in 'config.trainer.validation_interval' "
                    "of {}".format(config['trainer']['validation_interval']))

        if val_epoch <= 0:
            raise ValueError("Validation interval value ({}) of {} is invalid, expected "
                "value of more than zero (>0).".format("config.validator.val_epoch" 
                if set_in_validator else "config.trainer.validation_interval",
                val_epoch))

        return dict(check_val_every_n_epoch=val_epoch)

    @staticmethod
    def _trainer_args_set_seed(config):
        seed_cfg = None
        if 'seed' in config:
            warnings.warn("'config.seed' field is deprecated, please put the arguments in "
                "'config.trainer.seed'", DeprecationWarning)
            seed_cfg = config['seed']
        elif 'seed' in config['trainer']:
            seed_cfg = config['trainer']['seed']

        deterministic, benchmark = False, False
        if seed_cfg is not None:
            if isinstance(seed_cfg, int):   ## seed everything
                deterministic = True
                pl.seed_everything(seed_cfg)
                LOGGER.info("setting seed everything to '{}'".format(seed_cfg))
            elif isinstance(seed_cfg, dict):
                if 'cudnn' in seed_cfg:
                    warnings.warn("'seed.cudnn.*' argument in seed config is deprecated, move to "
                        "'seed.*' instead", DeprecationWarning)
                    seed_cfg.update(seed_cfg.pop('cudnn'))
                if 'benchmark' in seed_cfg:
                    benchmark = seed_cfg['benchmark']
                    LOGGER.info("setting torch.cudnn.benchmark to '{}'".format(benchmark))
                if 'deterministic' in seed_cfg:
                    deterministic = seed_cfg['deterministic']
                    LOGGER.info("setting cudnn.deterministic to '{}'".format(deterministic))
                if 'torch' in seed_cfg:
                    LOGGER.info("setting torch manual seed to '{}'".format(seed_cfg['torch']))
                    torch.manual_seed(seed_cfg['torch'])
                if 'numpy' in seed_cfg:
                    import numpy as np
                    np.random.seed(seed_cfg['numpy'])
                    LOGGER.info("setting numpy manual seed to {}".format(seed_cfg['numpy']))
            else:
                raise RuntimeError("Unknown seed config type of {} with value of {}".format(
                    type(seed_cfg), seed_cfg))
        return dict(deterministic=deterministic, benchmark=benchmark)

    @staticmethod
    def _trainer_args_accumulate_grad(config):
        accumulate = 1

        is_old_cfg_avail = ('driver' in config['trainer'] and 'args' in config['trainer']['driver'] and 
            'accumulation_step' in config['trainer']['driver']['args'])
        is_new_cfg_avail = ('accumulate_step' in config['trainer'])
        if is_old_cfg_avail:
            accumulate = config['trainer']['driver']['args']['accumulation_step']
        elif is_new_cfg_avail:
            accumulate = config['trainer']['accumulate_step']

        if isinstance(accumulate, int):
            is_value_valid = (accumulate > 0)
        elif isinstance(accumulate, dict):
            is_value_valid = all(x > 0 for x in accumulate.values())
        else:
            raise TypeError("Invalid type of {} in '{}' with value of {}, expected 'int' or 'dict'."
                .format(
                    type(accumulate), accumulate, ('config.driver.args.accumulation_step'
                    if is_old_cfg_avail else 'config.trainer.accumulate_step')
                ))
        if not is_value_valid:
            raise ValueError("Expected value of '{}' to be positive integer (>0), got {}"
                .format(
                    ('config.driver.args.accumulation_step' if is_old_cfg_avail 
                    else 'config.trainer.accumulate_step'), accumulate
                ))
        return dict(accumulate_grad_batches=accumulate)


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
    def load_config(config):
        if isinstance(config, (str, Path)):
            config = load_config(config)
        config = EasyDict(config)
        return config

    @staticmethod
    def _check_experiment_config(config):
        check_result = check_config(config, 'train')
        if not check_result.valid:
            raise RuntimeError("config file is not valid for training:\n{}".format(check_result))

        val_check_result = check_config(config, 'validate')
        if not val_check_result.valid:
            LOGGER.warning("config file is not valid for validation, validation step will be "
                "skipped:\n{}".format(val_check_result))

    @staticmethod
    def _dump_config(config, dir):
        fpath = Path(dir).joinpath("config.yml")
        fpath.parent.mkdir(parents=True, exist_ok=True)
        config = easydict_to_dict(config)
        with fpath.open('w') as f:
            yaml.dump(config, f, yaml.Dumper, sort_keys=False)
        return fpath

    @staticmethod
    def _handle_resume_checkpoint(config, resume):
        checkpoint, state_dict = None, None
        ckpt_path = None
        if resume or ('checkpoint' in config and config['checkpoint'] is not None):
            if resume and ('checkpoint' not in config or config['checkpoint'] is None):
                raise RuntimeError("You specify to resume but 'checkpoint' is not configured "
                    "in the config file. Please specify 'checkpoint' option in the top level "
                    "of your config file pointing to model path used for resume.")

            ckpt_path = Path(config.checkpoint)
            if (resume and ckpt_path.exists()) or ckpt_path.exists():
                checkpoint = torch.load(config.checkpoint, map_location=torch.device('cpu'))
                state_dict = checkpoint['state_dict']
            elif resume and not ckpt_path.exists():
                raise RuntimeError("Checkpoint path of {} is not exist, make sure to specify "
                    "the path properly".format(str(ckpt_path)))

            if resume:
                model_config = EasyDict(checkpoint['config'])
                if config.model.name != model_config.model.name:
                    raise RuntimeError("Model name configuration specified in config file ({}) is not "
                        "the same as saved in model checkpoint ({}).".format(config.model.name,
                        model_config.model.name))
                if config.model.network_args != model_config.model.network_args:
                    raise RuntimeError("'network_args' configuration specified in config file ({}) is "
                        "not the same as saved in model checkpoint ({}).".format(config.model.network_args, 
                        model_config.model.network_args))

                if 'dataset' in config and 'train' in config.dataset and 'name' in config.dataset.train:
                    cfg_dataset_name = config.dataset.train.name
                elif 'dataset' in config and 'train' in config.dataset and 'dataset' in config.dataset.train:
                    cfg_dataset_name = config.dataset.train.dataset
                else:
                    raise RuntimeError("dataset name is not found in config. Please specify in "
                        "'config.dataset.train.name'.")
                model_dataset_name = None
                if 'name' in model_config.dataset.train:
                    model_dataset_name = model_config.dataset.train.name
                elif 'dataset' in model_config.dataset.train:
                    model_dataset_name = model_config.dataset.train.dataset
                if cfg_dataset_name != model_dataset_name:
                    raise RuntimeError("Dataset specified in config file ({}) is not the same as saved "
                        "in model checkpoint ({}).".format(cfg_dataset_name, model_dataset_name))
            else:
                ckpt_path = None
        return ckpt_path, state_dict

    @staticmethod
    def _format_experiment_dir(config):
        base_exp_dir = Path('.').joinpath("experiments", "outputs")
        if 'output_directory' in config and config['output_directory']:
            base_exp_dir = Path(config['output_directory'])
        exp_dir = config['experiment_name']
        return base_exp_dir.joinpath(exp_dir)

    @staticmethod
    def _copy_final_checkpoint(trainer, config, experiment_dir, hypopt):
        if hypopt:
            return ""

        ckpt_last_callback = [c for c in trainer.checkpoint_callbacks
            if c.monitor is None and not hasattr(c, "save_epoch")]
        if len(ckpt_last_callback) == 0:
            return ""

        ckpt_last_callback = ckpt_last_callback[0]
        if ckpt_last_callback.best_model_path:
            last_fpath = Path(ckpt_last_callback.best_model_path)
            fname = f"{config['experiment_name']}.pth"
            ## copy weight to experiment dir
            final_fpath = Path(experiment_dir).joinpath(fname)
            shutil.copy(last_fpath, str(final_fpath))
            ## rename last epoch filename
            shutil.move(last_fpath, last_fpath.parent.joinpath(fname))
            return str(final_fpath)
