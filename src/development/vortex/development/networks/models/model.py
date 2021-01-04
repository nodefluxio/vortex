import torch
import pytorch_lightning as pl

from easydict import EasyDict
from abc import abstractmethod
from typing import Union
from copy import deepcopy
from torch.optim import Optimizer

from vortex.development.core.engine.trainer import lr_scheduler


class ModelBase(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.backbone = None
        self.postproces = None
        self.criterion = None
        self.preprocess = None

        ## will be set by vortex
        self.config: EasyDict = None
        self.class_names: list = None

        self._lr = 0.

    @torch.no_grad()
    @abstractmethod
    def predict(self, *args, **kwargs):
        """Predict function, called in inference
        includes preprocess, and postprocess.
        Expected:
        - input -> image in tensor/numpy with dim (NHWC) [N: batch, H: height, W: width, C: channel]
        - ouput -> list of tensors each member of the list corresponds
            to prediction of the image, or a tensor of dim (NxP) [N:batch, P: prediction]
        """
        pass

    def configure_optimizers(self):
        """Create optimizer and lr_scheduler
        """
        optimizer = create_optimizer(self.config, self.optimizer_param_groups)
        if 'lr_scheduler' in self.config['trainer']:
            cfg = create_scheduler(self.config, optimizer)
            cfg.update(dict(optimizer=optimizer))
            self._lr = cfg['lr_scheduler'].get_last_lr()[0]
        else:
            cfg = optimizer
            self._lr = optimizer.param_groups[0]['lr']
        return cfg

    def get_lr(self):
        assert self.trainer is not None, "trainer object is not yet defined, pass " \
            "this model to trainer first befor calling 'get_lr'"

        if len(self.trainer.lr_schedulers) > 0:
            lr = self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0]
        else:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self._lr = lr
        return lr

    @property
    def optimizer_param_groups(self):
        return self.parameters()

    @property
    def collate_fn(self):
        return None

    @property
    def lr(self):
        return self._lr

    @property
    @abstractmethod
    def output_format(self):
        pass

    @property
    @abstractmethod
    def available_metrics(self) -> Union[dict, list]:
        """explain all available metrics and the strategy for 
        getting the best value of it ('max' or 'min') as dict.
        or just return all available metrics as list but the
        strategy will be infered by the name of the metrics,
        if the name contains 'loss', the strategy is 'min'
        else use 'max'.

        example:
            {
                'val_loss': 'min',
                'accuracy': 'max'
            }
        or:
            ['val_loss', 'accuracy']
        """
        pass

    def on_export_start(self, exporter):
        pass

    def on_export_end(self, exporter, exported_model):
        pass


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
