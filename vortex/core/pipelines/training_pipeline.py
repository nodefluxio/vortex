import os
import shutil
import pytz
import warnings

from typing import Union
from pathlib import Path
from datetime import datetime
from copy import copy
from tqdm import tqdm
from easydict import EasyDict

import torch
import numpy as np
import comet_ml

from vortex.core.factory import (
    create_model,create_dataset,
    create_dataloader,
    create_experiment_logger
)
from vortex.utils.common import check_and_create_output_dir
from vortex.utils.parser import check_config
from vortex.core.pipelines.base_pipeline import BasePipeline
from vortex.core import engine

__all__ = ['TrainingPipeline']

def _set_seed(config : EasyDict):
    """Set pytorch and numpy seed https://pytorch.org/docs/stable/notes/randomness.html

    Args:
        config (EasyDict): dictionary parsed from Vortex experiment file from field 'seed'
    """
    try:
        seed = config.torch
        torch.manual_seed(seed)
        warnings.warn('setting torch manual seed to %s' % seed)
    except AttributeError:
        pass
    try:
        seed = config.numpy
        np.random.seed(config.numpy)
        warnings.warn('setting numpy manual seed to %s' % seed)
    except AttributeError:
        pass
    try:
        cudnn_deterministic = config.cudnn.deterministic
        torch.backends.cudnn.deterministic = cudnn_deterministic
        warnings.warn('setting cudnn.deterministic to %s' %
                      cudnn_deterministic)
    except AttributeError:
        pass
    try:
        cudnn_benchmark = config.cudnn.benchmark
        torch.backends.cudnn.benchmark = cudnn_benchmark
        warnings.warn('setting cudnn.benchmark to %s' % cudnn_benchmark)
    except AttributeError:
        pass

class TrainingPipeline(BasePipeline):
    """Vortex Training Pipeline API
    """

    def __init__(self,
                 config:EasyDict,
                 config_path: Union[str,Path,None] = None,
                 hypopt: bool = False,
                 resume: bool = False):
        """Class initialization

        Args:
            config (EasyDict): dictionary parsed from Vortex experiment file
            config_path (Union[str,Path,None], optional): path to experiment file. 
                Need to be provided for backup **experiment file**. 
                Defaults to None.
            hypopt (bool, optional): flag for hypopt, disable several pipeline process. 
                Defaults to False.
            resume (bool, optional): flag to resume training. 
                Defaults to False.

        Raises:
            Exception: raise undocumented error if exist

        Example:
            ```python
            from vortex.utils.parser import load_config
            from vortex.core.pipelines import TrainingPipeline
            
            # Parse config
            config_path = 'experiments/config/example.yml'
            config = load_config(config_path)
            train_executor = TrainingPipeline(config=config,
                                              config_path=config_path,
                                              hypopt=False)
            ```
        """

        self.start_epoch = 0
        state_dict = None
        if resume or 'checkpoint' in config:
            if 'checkpoint' not in config:
                raise RuntimeError("You specify to resume but 'checkpoint' is not configured "
                    "in the config file. Please specify 'checkpoint' option in the top level "
                    "of your config file pointing to model path used for resume.")
            checkpoint = torch.load(config.checkpoint)
            state_dict = checkpoint['state_dict']

            if resume:
                self.start_epoch = checkpoint['epoch']
                model_config = EasyDict(checkpoint['config'])
                if config.model.name != model_config.model.name:
                    raise RuntimeError("Model name configuration specified in config file ({}) is not "
                        "the same as saved in model checkpoint ({}).".format(config.model.name,
                        model_config.model.name))
                if config.model.network_args != model_config.model.network_args:
                    raise RuntimeError("'network_args' configuration specified in config file ({}) is "
                        "not the same as saved in model checkpoint ({}).".format(config.model.network_args, 
                        model_config.model.network_args))
                if config.dataset.train.dataset != model_config.dataset.train.dataset:
                    raise RuntimeError("Dataset specified in config file ({}) is not the same as saved "
                        "in model checkpoint ({}).".format(config.dataset.train.dataset, 
                        model_config.dataset.train.dataset))

                if ('n_classes' in config.model.network_args and 
                        (config.model.network_args.n_classes != model_config.model.network_args.n_classes)):
                    raise RuntimeError("Number of classes configuration specified in config file ({}) "
                        "is not the same as saved in model checkpoint ({}).".format(
                        config.model.network_args.n_classes, model_config.model.network_args.n_classes))

        self.config = config
        self.hypopt = hypopt

        # Check experiment config validity
        self._check_experiment_config(config)

        if not self.hypopt:
            # Create experiment logger
            self.experiment_logger = create_experiment_logger(config)

            # Output directory creation
            # If config_path is provided, it will duplicate the experiment file into the run directory
            self.experiment_directory,self.run_directory=check_and_create_output_dir(config,
                                                                                     self.experiment_logger,
                                                                                     config_path)

            # Create local experiments run log file
            self._create_local_runs_log(self.config,
                                        self.experiment_logger,
                                        self.experiment_directory,
                                        self.run_directory)
        else:
            self.experiment_logger=None

        # Training components creation
        if 'device' in config:
            self.device = config.device
        elif 'device' in config.trainer:
            self.device = config.trainer.device
        else:
            raise RuntimeError("'device' field not found in config. Please specify properly in main level.")
        self.model_components = create_model(model_config=config.model, state_dict=state_dict)
        self.model_components.network = self.model_components.network.to(self.device)
        self.criterion = self.model_components.loss.to(self.device)

        if 'dataloader' in config:
            dataloader_config = config.dataloader
        elif 'dataloader' in config.dataset:
            dataloader_config = config.dataset.dataloader
        else:
            raise RuntimeError("Dataloader config field not found in config.")

        augment_config = None
        if 'augmentations' in config:
            augment_config = config.augmentations
        elif 'augmentations' in config.dataset:
            augment_config = config.dataset.augmentations

        self.dataloader = create_dataloader(dataloader_config=dataloader_config,
                                            dataset_config=config.dataset,
                                            preprocess_config=config.model.preprocess_args,
                                            train_augment_config=augment_config,
                                            collate_fn=self.model_components.collate_fn,
                                            stage='train')
        self.trainer = engine.create_trainer(
            config.trainer, criterion=self.criterion,
            model=self.model_components.network,
            experiment_logger=self.experiment_logger,
        )
        if resume:
            self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
            if self.trainer.scheduler is not None:
                self.trainer.scheduler.load_state_dict(checkpoint["scheduler_state"])

        # Validation components creation
        try:
            val_dataset = create_dataset(config.dataset, config.model.preprocess_args, stage='validate')
            ## use same batch-size as training by default
            validation_args = EasyDict({'batch_size' : self.dataloader.batch_size})
            validation_args.update(config.trainer.validation.args)
            self.validator = engine.create_validator(
                self.model_components, 
                val_dataset, validation_args, 
                device=self.device
            )
            self.val_epoch = config.trainer.validation.val_epoch
            self.valid_for_validation = True
        except AttributeError as e:
            warnings.warn('validation step not properly configured, will be skipped')
            self.valid_for_validation = False
        except Exception as e:
            raise Exception(str(e))

        # Reproducibility settings check
        if hasattr(config, 'seed') :
            _set_seed(config.seed)

        if not self.hypopt:
            print("\nexperiment directory:", self.run_directory)
        self._has_cls_names = hasattr(self.dataloader.dataset, "class_names")

    def run(self,
            save_model : bool = True) -> EasyDict:
        """Execute training pipeline

        Args:
            save_model (bool, optional): dump model's checkpoint. Defaults to True.

        Returns:
            EasyDict: dictionary containing loss, val results and learning rates history

        Example:
            ```python
            train_executor = TrainingPipeline(config=config,
                                              config_path=config_path,
                                              hypopt=False)
            outputs = train_executor.run()
            ```
        """
        # Do training process
        val_metrics, val_results = [], None
        epoch_losses = []
        learning_rates = []
        for epoch in tqdm(range(self.start_epoch, self.config.trainer.epoch), desc="EPOCH",
                          total=self.config.trainer.epoch, initial=self.start_epoch, 
                          dynamic_ncols=True):
            loss, lr = self.trainer(self.dataloader, epoch)
            epoch_losses.append(loss.item())
            learning_rates.append(lr)
            print('epoch %s loss : %s with lr : %s' % (epoch, loss.item(), lr))

            # Experiment Logging
            metrics_log = EasyDict({
                'epoch' : epoch,
                'epoch_loss' : loss.item(),
                'epoch_lr' : lr
            })

            # Disable several training features for hyperparameter optimization
            if not self.hypopt:
                self.experiment_logger.log_on_epoch_update(metrics_log)

            # Do validation process if configured
            if self.valid_for_validation and ((epoch+1) % self.val_epoch == 0):
                assert(self.validator.predictor.model is self.model_components.network)
                val_results = self.validator()
                # val_results = {"accuracy": 0.3246}
                if 'pr_curves' in val_results :
                    val_results.pop('pr_curves')
                val_metrics.append(val_results)

                # Experiment Logging
                metrics_log = EasyDict({
                    'epoch' : epoch
                })

                # Assuming val_results type is dict
                metrics_log.update(val_results)

                # Disable several training features for hyperparameter optimization
                if not self.hypopt:
                    self.experiment_logger.log_on_validation_result(metrics_log)

                # logger.log_metrics(val_results, step=epoch)
                print('epoch %s validation : %s' % (epoch, ', '.join(['{}:{:.4f}'.format(key, value) for key, value in val_results.items()])))

            # Save on several epoch
            if ((epoch+1) % self.config.trainer.save_epoch == 0) and (save_model):
                metrics = None
                if self.valid_for_validation:
                    metrics = val_results
                model_fname = "{}-epoch-{}.pth".format(self.config.experiment_name, epoch)
                self._save_checkpoint(epoch, metrics=metrics, filename=model_fname)

                # Experiment Logging
                # Disabled on hyperparameter optimization
                if not self.hypopt:
                    file_log = EasyDict({
                        'epoch' : epoch,
                        'model_path' : model_fname
                    })
                    self.experiment_logger.log_on_model_save(file_log)

        # Save final weights on after all epochs finished
        if save_model:
            metrics = None
            if self.valid_for_validation:
                metrics = val_results
            model_fname = "{}.pth".format(self.config.experiment_name)
            model_path = self._save_checkpoint(epoch, metrics=metrics, filename=model_fname)
            # Copy final weight from runs directory to experiment directory
            shutil.copy(model_path, Path(self.experiment_directory))

            # Experiment Logging
            # Disabled on hyperparameter optimization
            if not self.hypopt:
                file_log = EasyDict({
                    'epoch' : epoch,
                    'model_path' : model_path
                })
                self.experiment_logger.log_on_model_save(file_log)

        return EasyDict({
            'epoch_losses' : epoch_losses, 
            'val_metrics' : val_metrics, 
            'learning_rates' : learning_rates
        })

    def _check_experiment_config(self,config : EasyDict):
        """Function to check whether configuration is valid for training

        Args:
            config (EasyDict): dictionary parsed from Vortex experiment file

        Raises:
            RuntimeError: raise error if configuration is not valid
        """
        check_result = check_config(config, 'train')
        if not check_result.valid:
            raise RuntimeError("invalid config : %s" % str(check_result))
        val_check_result = check_config(config, 'validate')
        if not val_check_result.valid:
            warnings.warn('this config file is not valid for validation, validation step will be "\
                "skipped: \n%s' % str(val_check_result))

    def _create_local_runs_log(self,
                              config : EasyDict,
                              experiment_logger,
                              experiment_directory : Union[str,Path],
                              run_directory : Union[str,Path]):
        """Log experiment attempt to 'local_runs.log'

        Args:
            config (EasyDict): dictionary parsed from Vortex experiment file
            experiment_logger : ExperimentLogger object
            experiment_directory (Union[str,Path]): path to experiment directory
            run_directory (Union[str,Path]): path to run directory
        """
        # Log experiments run on local experiment runs log file
        utc_now = pytz.utc.localize(datetime.utcnow())
        if not 'pytz_timezone' in config.logging:
            pst_now = utc_now.astimezone(pytz.timezone("Asia/Jakarta"))
        else:
            pst_now = utc_now.astimezone(pytz.timezone(config.logging.pytz_timezone))
        date_time = pst_now.strftime("%m/%d/%Y, %H:%M:%S")
        logging_provider = 'None' if config.logging == 'None' else config.logging.module
        log_file = Path('experiments/local_runs.log')
        if log_file.exists():
            mode='r+'
        else:
            mode='a+'
        with open(log_file,mode) as f:
            if mode=='r+':
                old_content = f.read()
                f.seek(0,0)
            f.write('#'*100+'\n')
            f.write('Timestamp : %s\n'%date_time)
            f.write('Experiment Name : %s\n'%config.experiment_name)
            f.write('Base Experiment Output Path : %s\n'%experiment_directory)
            f.write('Experiment Run Output Path : %s\n'%run_directory)
            f.write('Logging Provider : %s\n'%logging_provider)
            if 'log_url' in experiment_logger.__dict__.keys():
                f.write('Experiment Log URL : %s\n'%experiment_logger.log_url)
            f.write('#'*100+'\n')
            f.write('\n')
            if mode=='r+':
                f.write(old_content)

    def _save_checkpoint(self, epoch, metrics=None, filename=None):
        checkpoint = {
            "epoch": epoch+1, 
            "config": self.config,
            "state_dict": self.model_components.network.state_dict(),
            "optimizer_state": self.trainer.optimizer.state_dict(),
        }
        if metrics is not None:
            checkpoint["metrics"] = metrics
        if self._has_cls_names and self.dataloader.dataset.class_names is not None:
            checkpoint["class_names"] = self.dataloader.dataset.class_names
        if self.trainer.scheduler is not None:
            checkpoint["scheduler_state"] = self.trainer.scheduler.state_dict()

        filedir = self.run_directory
        if filename is None:
            filename = "{}.pth".format(self.config.experiment_name)
        else:
            filename = Path(filename)
            if str(filename.parent) != ".":
                filedir = filename.parent
                filename = Path(filename.name)
            if filename.suffix == "" or filename.suffix != ".pth":
                if filename.suffix != ".pth":
                     warnings.warn("filename for save checkpoint ({}) does not have "
                        ".pth extension, overriding it to .pth".format(str(filename)))
                filename = filename.with_suffix(".pth")

        filepath =  str(filedir.joinpath(filename))
        torch.save(checkpoint, filepath)
        return filepath
