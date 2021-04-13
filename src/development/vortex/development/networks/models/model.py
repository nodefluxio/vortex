import torch
import pytorch_lightning as pl

from easydict import EasyDict
from abc import abstractmethod
from typing import List, Union, Callable

from vortex.development.utils import create_optimizer, create_scheduler


class ModelBase(pl.LightningModule):
    def __init__(self):
        super().__init__()

        ## will be set by vortex
        self.config: EasyDict = None
        self.class_names: list = None

        self._lr = 0.

        ## TODO: add pretrained backbone properties (normalization)


    @torch.no_grad()
    @abstractmethod
    def predict(self, *args, **kwargs) -> Union[List[torch.Tensor], torch.Tensor]:
        """Predict function, called in inference
        includes preprocess, and postprocess.
        Expected:
            - input -> image in tensor/numpy with dim (NHWC) [N: batch, H: height, W: width, C: channel]
            - ouput -> list of tensors each member of the list corresponds to prediction of the image, or a tensor of dim (NxP) [N:batch, P: prediction]
        """
        pass

    def configure_optimizers(self):
        """Create optimizer and lr_scheduler
        """
        optimizer = create_optimizer(self.config, self.optimizer_param_groups)
        if 'lr_scheduler' in self.config['trainer'] and self.config['trainer']['lr_scheduler'] is not None:
            cfg = create_scheduler(self.config, optimizer)
            cfg.update(dict(optimizer=optimizer))
            self._lr = cfg['lr_scheduler'].get_last_lr()[0]
        else:
            cfg = optimizer
            self._lr = optimizer.param_groups[0]['lr']
        return cfg

    def get_lr(self) -> Union[float, torch.Tensor]:
        assert self.trainer is not None, "trainer object is not yet defined, pass " \
            "this model to trainer first befor calling 'get_lr'"

        if len(self.trainer.lr_schedulers) > 0:
            lr = self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0]
        else:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self._lr = lr
        return lr

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_step_end(self, outputs):
        return self.validation_step_end(outputs)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    @property
    def optimizer_param_groups(self):
        return self.parameters()

    @property
    def collate_fn(self) -> Union[Callable]:
        return None

    @property
    def lr(self):
        return self._lr

    @property
    @abstractmethod
    def output_format(self):
        """Define the format ('indices' and 'axis' for each output) to
        extract the output for single batch.
        """
        pass

    @property
    @abstractmethod
    def available_metrics(self) -> Union[dict, List[str]]:
        """explain all available metrics and the strategy for 
        getting the best value of it ('max' or 'min') as dict.
        or just return all available metrics as list but the
        strategy will be infered by the name of the metrics,
        if the name contains 'loss', the strategy is 'min'
        else use 'max'.

        example:

        .. code-block:: python

            {
                'val_loss': 'min',
                'accuracy': 'max'
            }

        or:

        .. code-block:: bash

            ['val_loss', 'accuracy']
        """
        pass

    def on_export_start(self, exporter):
        """This method will be called at the start of export
        session.

        Args:
            exporter: exporter that is exporting this model
        """
        pass

    def on_export_end(self, exporter, exported_model):
        """This method will be called after the model is exported

        Args:
            exporter: exporter that is exporting this model
            exported_model: exported model by the exporter
        """
        pass
