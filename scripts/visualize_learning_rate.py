"""Utility to visualize learning rates from the vortex config

This script is used to visualize the value of the resulting
learning rates from learning rate scheduler in the configuration 
files. It is required for the config file to have scheduler 
configuration in `trainer.lr_scheduler`.

To get started using this script, try:
```
$ python scripts/visualize_learning_rate.py --config 
  experiments/configs/efficientnet_b0_classification_cifar10.yml
  --epochs 100
```

For the more details argument you can use, see the `--help` argument.
"""

import sys
from pathlib import Path
from matplotlib.pyplot import plot
import math
from easydict import EasyDict

from torch import optim
sys.path.append(str(Path(__file__).parents[1].joinpath('src', 'development')))

import argparse
import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
from vortex.development.utils.parser import load_config
from vortex.development.core.engine.trainer.base_trainer import BaseTrainer
from vortex.development.core.factory import create_dataloader
warnings.resetwarnings()


def calculate_lr(lr_config, epochs=100, optimizer_config=None,accumulation_step=1,steps_per_epoch=100):
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
    # assert hasattr(lr_scheduler, sch_method), "unsupported lr_scheduler {}".format(sch_method)
    # kwargs.update({'optimizer': optimizer})
    # scheduler = getattr(lr_scheduler, sch_method)(**kwargs)
    scheduler = BaseTrainer.create_scheduler(lr_config,optimizer)

    print("Visualizing {} lr scheduler for {} epoch".format(sch_method, epochs))
    lr_data = []
    for ep in range(epochs):

        for i in range(steps_per_epoch):

            if (i+1) % accumulation_step == 0:
                optimizer.step()
                BaseTrainer.apply_scheduler_step(scheduler,
                                                epoch = ep,
                                                step = i,
                                                steps_per_epoch = steps_per_epoch,
                                                accumulation_step = accumulation_step)
                optimizer.zero_grad()
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
    parser.add_argument('--accumulation_step', required=False, type=int, 
        help="number of gradient accumulation step to be calculated, override accumulation step in experiment file")
    parser.add_argument('--steps_per_epoch', required=False, type=int, 
        help="number of steps per epoch to be calculated, override dataloader len in experiment file")
        
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

    # Get accumulation step
    accumulation_step = 1
    
    ## Try to get accumulation_step from argparse or experiment config
    if args.accumulation_step is not None:
        accumulation_step = args.accumulation_step
    else:
        try:
            accumulation_step = config.trainer.driver.args.accumulation_step
        except:
            pass

    # Get steps per epoch

    steps_per_epoch = 100

    ## Try to get accumulation_step from argparse or experiment config
    if args.steps_per_epoch is not None:
        steps_per_epoch = args.steps_per_epoch
    else:
        try:
            dataloader = create_dataloader(dataloader_config=config.dataloader,
                                            dataset_config=config.dataset,
                                            preprocess_config = config.model.preprocess_args,
                                            collate_fn=None)
            steps_per_epoch = len(dataloader)
        except:
            pass
    

    lrs = calculate_lr(lr_scheduler_cfg, 
                        epochs=epochs, 
                        optimizer_config=optim_cfg, 
                        accumulation_step=accumulation_step,
                        steps_per_epoch=steps_per_epoch)

    

    plt.plot(lrs)
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.ylabel("learning rate")
    plt.xlabel("epoch")
    plt.title("{} Learning Rate in {} epoch".format(lr_scheduler_cfg['module'], epochs))
    plt.show()
