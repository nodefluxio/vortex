import torch
import shutil

from torch.optim import Optimizer
from copy import deepcopy
from pathlib import Path
from easydict import EasyDict
from typing import Union

from . import lr_scheduler


def create_optimizer(config, param_groups) -> Optimizer:
    """create optimizer from vortex config
    """
    optim_cfg = config['trainer']['optimizer']
    if 'method' in optim_cfg:
        module = optim_cfg['method']
    else:
        module = optim_cfg['module']
    kwargs = deepcopy(optim_cfg['args'])
    kwargs.update(dict(params=param_groups))
    if not hasattr(torch.optim, module):
        raise RuntimeError("Optimizer module '{}' is not available, see "
            "https://pytorch.org/docs/stable/optim.html#algorithms for "
            "all available optimizer modules".format(module))
    optim = getattr(torch.optim, module)(**kwargs)
    return optim

def create_scheduler(config, optimizer) -> dict:
    """create scheduler and the PL config as dict from vortex config
    """
    scheduler_cfg = config['trainer']['lr_scheduler']
    if 'method' in scheduler_cfg:
        module = scheduler_cfg['method']
    else:
        module = scheduler_cfg['module']
    if not hasattr(lr_scheduler, module):
        raise RuntimeError("LR Scheduler module '{}' is not available")
    interval = "epoch" if module in lr_scheduler.step_update_map['epoch_update'] \
                else "step"

    freq = 1
    if 'frequency' in scheduler_cfg:
        freq = scheduler_cfg['frequency']
    monitor = None
    if 'monitor' in scheduler_cfg:
        monitor = scheduler_cfg['monitor']

    kwargs = scheduler_cfg['args']
    kwargs.update(dict(optimizer=optimizer))
    scheduler = getattr(lr_scheduler, module)(**kwargs)
    ret = {
        'lr_scheduler': scheduler,
        'interval': interval,
        'frequency': freq,
        'strict': True,
    }
    if monitor:
        ret.update(dict(monitor=monitor))
    return ret


def check_and_create_output_dir(config : EasyDict,
                                experiment_logger = None,
                                config_path : Union[str,Path,None] = None):

    # Set base output directory
    base_output_directory = Path('experiments/outputs')
    if 'output_directory' in config:
        base_output_directory = Path(config.output_directory)

    # Set experiment directory
    experiment_directory = Path(base_output_directory/config.experiment_name)
    if not experiment_directory.exists():
        experiment_directory.mkdir(exist_ok=True, parents=True)

    # Set run directory
    run_directory = None
    if experiment_logger:
        run_directory=Path(experiment_directory/experiment_logger.run_key)
        if not run_directory.exists():
            run_directory.mkdir(exist_ok=True, parents=True)
        # Duplicate experiment config if specified to run directory
        if config_path:
            shutil.copy(config_path,str(run_directory/'config.yml'))

    return experiment_directory,run_directory