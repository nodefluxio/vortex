import sys
from pathlib import Path
from matplotlib.pyplot import plot

from torch import optim
sys.path.append(str(Path(__file__).parents[1].joinpath('src', 'development')))

import argparse
import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
from vortex.development.utils.parser import load_config
from vortex.development.core.engine.trainer import lr_scheduler
warnings.resetwarnings()


def calculate_lr(lr_config, epochs=100, optimizer_config=None):
    ## optimizer first
    if optimizer_config is None:
        optimizer_config = { ## just put a dummy optimizer
            'module': 'SGD',
            'args': {
                'lr': 0.01,
                'momentum': 0.95,
                'weight_decay': 0.0001,
            }
        }

    if 'method' in optimizer_config and not 'module' in optimizer_config:
        optimizer_config.update({'module': optimizer_config['method']})
    opt_method, kwargs = optimizer_config['module'], optimizer_config['args']
    if opt_method.startswith('optim.'):
        opt_method = opt_method.replace('optim.','')
    assert hasattr(optim, opt_method), "unsupported optimizer {}".format(opt_method)
    kwargs.update({'params' : torch.nn.Linear(2, 1).parameters()})  ## dummy model
    optimizer = getattr(optim, opt_method)(**kwargs)

    ## lr scheduler
    if 'method' in lr_config and not 'module' in lr_config:
        lr_config.update({'module': lr_config['method']})
    sch_method, kwargs = lr_config['module'], lr_config['args']
    assert hasattr(lr_scheduler, sch_method), "unsupported lr_scheduler {}".format(sch_method)
    kwargs.update({'optimizer': optimizer})
    scheduler = getattr(lr_scheduler, sch_method)(**kwargs)

    print("Visualizing {} lr scheduler for {} epoch".format(sch_method, epochs))
    lr_data = []
    for ep in range(epochs):
        optimizer.step()
        try:
            scheduler.step()
        except:
            scheduler.step(ep)
        lr_data.append(scheduler.get_last_lr())
    return np.array(lr_data)


def visualize(learning_rates):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize learning rate scheduler")
    parser.add_argument('CFG', type=str, nargs='?', help="experiment configuration file path (.yml)")
    parser.add_argument('-c', '--config', required=False, type=str,
        help="experiment configuration file path (choose either one of this or positional argument)")
    parser.add_argument('--epochs', required=False, type=int, 
        help="number of epoch of learning rate to be calculated, override epoxh in experiment file")
    args = parser.parse_args()

    if args.config is None and args.CFG is None:
        raise RuntimeError("config path containing learning rate definition is required, "
            "please specify them in the positional arguments or use '--config' argument.")
    if args.config and args.CFG and args.config != args.CFG:
        warnings.warn("both positional argument and '--config' argument is specified, "
            "using the positional argument value of {}".format(args.CFG))
    if args.CFG is not None:
        args.config = args.CFG
    config = load_config(args.config)

    lr_scheduler_cfg = None
    if 'scheduler' in config.trainer and not 'lr_scheduler' in config.trainer:
        lr_scheduler_cfg = config.trainer.scheduler
    elif 'lr_scheduler' in config.trainer:
        lr_scheduler_cfg = config.trainer.lr_scheduler

    if lr_scheduler_cfg is None or lr_scheduler_cfg == {}:
        raise RuntimeError("learning rate scheduler configuration not found in config file, "
            "please specify it in 'config.trainer.lr_scheduler'.")

    optim_cfg = None
    if 'optimizer' in config.trainer and config.trainer.optimizer != {}:
        optim_cfg = config.trainer.optimizer

    epochs = None
    if 'epoch' in config.trainer:
        epochs = int(config.trainer.epoch)
    if args.epochs is not None:
        epochs = args.epochs
    if epochs is None:
        warnings.warn("number of epochs is not set, using default value of 100. Specify the "
            "epochs value in the cofig file in 'config.trainer.epochs' or using '--epochs' argument.")
        epochs = 100

    lrs = calculate_lr(lr_scheduler_cfg, epochs=epochs, optimizer_config=optim_cfg)

    plt.plot(lrs)
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.ylabel("learning rate")
    plt.xlabel("epoch")
    plt.title("{} Learning Rate in {} epoch".format(lr_scheduler_cfg['module'], epochs))
    plt.show()
