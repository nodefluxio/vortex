
import torch
from tqdm import tqdm
from easydict import EasyDict
from typing import Tuple, List, Union, Type, Any, Dict

from vortex.development.core.engine.trainer.base_trainer import BaseTrainer

class DefaultTrainer(BaseTrainer):
    def __init__(self, accumulation_step: int = 1, *args, **kwargs):
        super(DefaultTrainer, self).__init__(*args, **kwargs)
        assert accumulation_step >= 1, \
            "accumulation_step should be >= 1, got {}".format(accumulation_step)
        self.accumulation_step = accumulation_step
        self.lr = self.optimizer.param_groups[0]['lr']

    def train(self, dataloader, epoch):
        """
        default train
        """
        epoch_loss, step_loss = 0., 0.
        ## TODO : consider to move device deduction to BaseTrainer
        device = list(self.model.parameters())[0].device
        for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader),
                                         desc=" train", leave=False):
            if self.scheduler is not None:
                self.lr = self.scheduler.get_last_lr()[0]
            inputs = inputs.to(device)
            if isinstance(targets, torch.Tensor):
                targets = targets.to(device)
            elif isinstance(targets, (list, tuple)):
                if isinstance(targets[0], dict):
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                elif isinstance(targets[0], torch.Tensor):
                    targets = [t.to(device) for t in targets]

            preds = self.model(inputs)
            batch_loss = self.criterion(preds, targets)
            batch_loss.backward()
            epoch_loss += batch_loss.detach()
            step_loss += batch_loss.detach()
            if (i+1) % self.accumulation_step == 0:
                self.optimizer.step()
                if self.scheduler:
                    try:
                        self.scheduler.step()
                    except:
                        self.scheduler.step(epoch)
                self.optimizer.zero_grad()

                # Experiment Logging
                metrics_log = EasyDict({
                    'step' : self.global_step,
                    'step_loss' : step_loss.item()/self.accumulation_step,
                    'step_lr' : self.lr
                })

                if self.experiment_logger:
                    self.experiment_logger.log_on_step_update(metrics_log)

                self.global_step+=1
                step_loss = 0
        return (epoch_loss / len(dataloader)), self.lr