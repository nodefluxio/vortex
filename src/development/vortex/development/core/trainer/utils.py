import torch

from torch.optim import Optimizer
from copy import deepcopy

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
