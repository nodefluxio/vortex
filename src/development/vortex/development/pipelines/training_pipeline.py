import os
import shutil
import pytz
import warnings
import logging

from typing import Union
from pathlib import Path
from datetime import datetime
from easydict import EasyDict

import torch
import numpy as np
import enlighten

from vortex.development.utils.factory import (
    create_model,
    create_dataset,
    create_dataloader,
    create_experiment_logger
)
from vortex.development.utils.common import check_and_create_output_dir
from vortex.development.utils.parser import check_config
from vortex.development.pipelines.base_pipeline import BasePipeline

__all__ = ['TrainingPipeline']

logger = logging.getLogger(__name__)

def _set_seed(config : EasyDict):
    """Set pytorch and numpy seed https://pytorch.org/docs/stable/notes/randomness.html

    Args:
        config (EasyDict): dictionary parsed from Vortex experiment file from field 'seed'
    """
    try:
        seed = config.torch
        torch.manual_seed(seed)
        logger.info('setting torch manual seed to %s' % seed)
    except AttributeError:
        pass
    try:
        seed = config.numpy
        np.random.seed(config.numpy)
        logger.info('setting numpy manual seed to %s' % seed)
    except AttributeError:
        pass
    try:
        cudnn_deterministic = config.cudnn.deterministic
        torch.backends.cudnn.deterministic = cudnn_deterministic
        logger.info('setting cudnn.deterministic to %s' %
                      cudnn_deterministic)
    except AttributeError:
        pass
    try:
        cudnn_benchmark = config.cudnn.benchmark
        torch.backends.cudnn.benchmark = cudnn_benchmark
        logger.info('setting cudnn.benchmark to %s' % cudnn_benchmark)
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
            from vortex.development.utils.parser import load_config
            from vortex.development.core.pipelines import TrainingPipeline
            
            # Parse config
            config_path = 'experiments/config/example.yml'
            config = load_config(config_path)
            train_executor = TrainingPipeline(config=config,
                                              config_path=config_path,
                                              hypopt=False)
            ```
        """

        ## terminal output related manager
        self.ui_manager = enlighten.get_manager()

        ## title bar
        title_fmt = "Vortex{fill}TRAIN: {exp_name}{fill}{elapsed}"
        self.ui_manager.status_bar(
            status_format=title_fmt, 
            color="bold_underline_white",
            justify=enlighten.Justify.CENTER, 
            autorefresh=True, min_delay=0.5,
            exp_name=config.experiment_name
        )

        self.start_epoch = 0
        checkpoint, state_dict = None, None
        if resume or ('checkpoint' in config and config.checkpoint is not None):
            if 'checkpoint' not in config:
                raise RuntimeError("You specify to resume but 'checkpoint' is not configured "
                    "in the config file. Please specify 'checkpoint' option in the top level "
                    "of your config file pointing to model path used for resume.")
            if resume or os.path.exists(config.checkpoint):
                checkpoint = torch.load(config.checkpoint, map_location=torch.device('cpu'))
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

                if 'name' in config.dataset.train:
                    cfg_dataset_name = config.dataset.train.name
                elif 'dataset' in config.dataset.train:
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

        model_components = create_model(model_config=config.model, state_dict=state_dict)
        if not isinstance(model_components, EasyDict):
            model_components = EasyDict(model_components)
        # not working for easydict
        # model_components.setdefault('collate_fn',None)
        if not 'collate_fn' in model_components:
            model_components.collate_fn = None
        self.model_components = model_components

        self.model_components.network = self.model_components.network.to(self.device)
        self.criterion = self.model_components.loss.to(self.device)

        param_groups = None
        if 'param_groups' in self.model_components:
            param_groups = self.model_components.param_groups

        if 'dataloader' in config:
            dataloader_config = config.dataloader
        elif 'dataloader' in config.dataset:
            dataloader_config = config.dataset.dataloader
        else:
            raise RuntimeError("Dataloader config field not found in config.")

        self.dataloader = create_dataloader(dataloader_config=dataloader_config,
                                            dataset_config=config.dataset,
                                            preprocess_config=config.model.preprocess_args,
                                            collate_fn=self.model_components.collate_fn,
                                            stage='train')
        self.trainer = engine.create_trainer(
            config.trainer, criterion=self.criterion,
            model=self.model_components.network,
            experiment_logger=self.experiment_logger,
            param_groups=param_groups
        )
        if resume:
            self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
            if self.trainer.scheduler is not None:
                scheduler_args = self.config.trainer.lr_scheduler.args
                if isinstance(scheduler_args, dict):
                    for name, v in scheduler_args.items():
                        if name in checkpoint["scheduler_state"]:
                            checkpoint["scheduler_state"][name] = v
                self.trainer.scheduler.load_state_dict(checkpoint["scheduler_state"])

        has_save = False
        self.save_best_metrics, self.save_best_type = None, None
        self.best_metrics = None
        if 'save_best_metrics' in self.config.trainer and self.config.trainer.save_best_metrics is not None:
            has_save = self.config.trainer.save_best_metrics is not None
            self.save_best_metrics = self.config.trainer.save_best_metrics
            if not isinstance(self.save_best_metrics, (list, tuple)):
                self.save_best_metrics = [self.save_best_metrics]

            self.save_best_type = list({'loss' if m == 'loss' else 'val_metric' for m in self.save_best_metrics})
            self.best_metrics = {name: float('inf') if name == 'loss' else float('-inf') for name in self.save_best_metrics}
            if 'loss' in self.save_best_metrics:
                self.save_best_metrics.remove('loss')

            if resume:
                best_metrics_ckpt = checkpoint['best_metrics']
                if isinstance(best_metrics_ckpt, dict):
                    self.best_metrics.update(best_metrics_ckpt)

        self.save_epoch, self.save_last_epoch = None, None
        if 'save_epoch' in self.config.trainer and self.config.trainer.save_epoch is not None:
            self.save_epoch = self.config.trainer.save_epoch
            has_save = has_save or self.config.trainer.save_epoch is not None
        if not has_save:
            warnings.warn("No model checkpoint saving configuration is specified, the training would still "
                "work but will only save the last epoch model.\nYou can configure either one of "
                "'config.trainer.save_epoch' or 'config.trainer.save_best_metric")

        # Validation components creation
        try:
            if 'validator' in config:
                validator_cfg = config.validator
            elif 'device' in config.trainer:
                validator_cfg = config.trainer.validator
            else:
                raise RuntimeError("'validator' field not found in config. Please specify properly in main level.")

            val_dataset = create_dataset(config.dataset, config.model.preprocess_args, stage='validate')
            
            ## use same batch-size as training by default
            validation_args = EasyDict({'batch_size' : self.dataloader.batch_size})
            validation_args.update(validator_cfg.args)
            self.validator = engine.create_validator(
                self.model_components, 
                val_dataset, validation_args, 
                device=self.device
            )
            
            self.val_epoch = validator_cfg.val_epoch
            self.valid_for_validation = True
        except AttributeError as e:
            logger.warning('validation step not properly configured, will be skipped')
            self.valid_for_validation = False
        except Exception as e:
            raise Exception(str(e))

        # Reproducibility settings check
        if hasattr(config, 'seed') :
            _set_seed(config.seed)

        if not self.hypopt:
            print("\nExperiment directory:", self.run_directory)
        self._has_cls_names = hasattr(self.dataloader.dataset, "class_names")

    def run(self, save_model: bool = True) -> EasyDict:
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

        ## metric bar
        metric_format = "metrics    lr: {lr:.4g}  loss: {loss:.4g}  {metrics}{fill}"
        metric_stats = self.ui_manager.status_bar(
            status_format=metric_format,
            justify=enlighten.Justify.LEFT, color="white_on_gray20",
            position=1,
            loss=0.0, lr=0.0, metrics=""
        )
        self.ui_manager.status_bar(status_format="{fill}", position=2)

        epoch_bar_fmt = u'{desc}{desc_pad}{percentage:3.0f}%|{bar}| ' + \
                        u'{count:{len_total}d}/{total:d} ' + \
                        u'[{elapsed}<{eta}, {secs_per_iter:.2f}s/{unit}]'
        total_epoch = self.config.trainer.epoch
        epoch_pbar = self.ui_manager.counter(
            count=self.start_epoch, total=total_epoch,
            desc="Epoch:", unit="epoch",
            bar_format=epoch_bar_fmt,
            secs_per_iter=0.0
        )

        default_bar_fmt = '{desc}{desc_pad}{percentage:3.0f}%|{bar}| ' + \
                          '{count:{len_total}d}/{total:d} ' + \
                          '[{elapsed}<{eta}, {rate:.2f}{unit}/s]'

        ## Do training process
        val_metrics, val_results = [], None
        epoch_losses = []
        learning_rates = []
        for epoch in range(self.start_epoch, self.config.trainer.epoch):
            train_pbar = self.ui_manager.counter(
                total=len(self.dataloader), desc="  Training:", 
                unit='it', leave=False,
                bar_format=default_bar_fmt,
            )

            loss, lr = self.trainer(self.dataloader, epoch, train_pbar)
            epoch_losses.append(loss.item())
            learning_rates.append(lr)
            metric_stats.update(loss=loss.item(), lr=lr)

            # Experiment Logging, disable on hyperparameter optimization
            if not self.hypopt:
                metrics_log = EasyDict({
                    'epoch' : epoch,
                    'epoch_loss' : loss.item(),
                    'epoch_lr' : lr
                })
                self.experiment_logger.log_on_epoch_update(metrics_log)

            # Do validation process if configured
            if self.valid_for_validation and ((epoch+1) % self.val_epoch == 0):
                assert(self.validator.predictor.model is self.model_components.network)
                val_pbar = self.ui_manager.counter(
                    total=len(self.validator.dataset), desc='  Validating:', 
                    unit='it', leave=False,
                    bar_format=default_bar_fmt
                )

                val_results = self.validator(val_pbar)
                if 'pr_curves' in val_results:
                    val_results.pop('pr_curves')
                val_metrics.append(val_results)

                disp_metrics = {"val_loss": val_results["val_loss"]} if "val_loss" in val_results else {}
                disp_metrics.update({k: val_results[k] for k in list(val_results.keys())[:2]})
                disp_metrics = '  '.join("{}: {:.4g}".format(n, v) for n,v in disp_metrics.items())
                metric_stats.update(metrics=disp_metrics)

                # Experiment Logging, disable on hyperparameter optimization
                if not self.hypopt:
                    metrics_log = EasyDict(dict(epoch=epoch, **val_results))
                    self.experiment_logger.log_on_validation_result(metrics_log)

                # Drop val loss from metric
                val_results.pop('val_loss')

                if self.save_best_type and 'val_metric' in self.save_best_type and save_model:
                    for metric_name in self.save_best_metrics:
                        if not metric_name in val_results:
                            val_res_key = ", ".join(list(val_results.keys()))
                            raise RuntimeError("'save_best_metric' value of ({}) is not found in validation "
                                "result, choose either one of [{}]".format(metric_name, val_res_key))
                        if self.best_metrics[metric_name] < val_results[metric_name]:
                            self.best_metrics[metric_name] = val_results[metric_name]
                            model_fname = "{}-best-{}.pth".format(self.config.experiment_name, metric_name)
                            self._save_checkpoint(epoch, metrics=val_results, filename=model_fname)

            ## save on best loss
            if self.save_best_type and 'loss' in self.save_best_type and \
                    self.best_metrics['loss'] > loss.item() and save_model:
                self.best_metrics['loss'] = loss.item()
                model_fname = "{}-best-loss.pth".format(self.config.experiment_name)
                self._save_checkpoint(epoch, metrics=val_results, filename=model_fname)

            # Save on several epoch
            if self.save_epoch and ((epoch+1) % self.save_epoch == 0) and save_model:
                model_fname = "{}-epoch-{}.pth".format(self.config.experiment_name, epoch)
                self._save_checkpoint(epoch, metrics=val_results, filename=model_fname)

            if save_model: ## save this epoch
                model_fname = "{}-last.pth".format(self.config.experiment_name)
                model_path = self._save_checkpoint(epoch, metrics=val_results, filename=model_fname)
                if epoch == self.config.trainer.epoch-1:
                    ## Copy final weight from runs directory to experiment directory
                    final_path = self.experiment_directory.joinpath("{}.pth".format(self.config.experiment_name))
                    shutil.copy(model_path, final_path)
                    ## rename last epoch weight
                    shutil.move(model_path, model_path.replace('-last', ''))
            epoch_elapsed = epoch_pbar.elapsed / (epoch_pbar.count+1)
            epoch_pbar.update(secs_per_iter=epoch_elapsed)
        epoch_pbar.close()
        metric_stats.close()
        self.ui_manager.stop()

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
        self.config = config ## this is just a workaround for backward compatibility
        val_check_result = check_config(config, 'validate')
        if not val_check_result.valid:
            logger.warning('this config file is not valid for validation, validation step will be "\
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
        if not config.logging or (config.logging and not 'pytz_timezone' in config.logging):
            pst_now = utc_now.astimezone(pytz.timezone("Asia/Jakarta"))
        else:
            pst_now = utc_now.astimezone(pytz.timezone(config.logging.pytz_timezone))
        date_time = pst_now.strftime("%m/%d/%Y, %H:%M:%S")
        logging_provider = None if config.logging == None else config.logging.module
        log_file = Path('experiments/local_runs.log')
        if not os.path.isdir('experiments'):
            os.makedirs('experiments')
        if log_file.exists():
            mode='r+'
        else:
            mode='a+'
        with open(log_file,mode) as f:
            old_content = ""
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
        if not self.hypopt:
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
            if self.best_metrics is not None:
                checkpoint["best_metrics"] = self.best_metrics
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
                        logger.info("filename for save checkpoint ({}) does not have "
                            ".pth extension, overriding it to .pth".format(str(filename)))
                    filename = filename.with_suffix(".pth")

            filepath =  str(filedir.joinpath(filename))
            torch.save(checkpoint, filepath)

            file_log = EasyDict({
                'epoch' : epoch,
                'model_path' : filepath
            })
            self.experiment_logger.log_on_model_save(file_log)
            return filepath
        else:
            return None
