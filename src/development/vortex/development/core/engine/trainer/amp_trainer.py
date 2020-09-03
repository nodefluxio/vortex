from vortex.development.core.engine.trainer.base_trainer import BaseTrainer
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from easydict import EasyDict

class AMPTrainer(BaseTrainer):
    def __init__(self, accumulation_step: int = 1, *args, **kwargs):
        super(AMPTrainer, self).__init__(*args, **kwargs)
        
        # AMP trainer only supports model on cuda device
        self.device = list(self.model.parameters())[0].device
        assert self.device.type=='cuda', 'AMPTrainer can only be used on cuda device'

        assert accumulation_step >= 1, \
            "accumulation_step should be >= 1, got {}".format(accumulation_step)
        self.accumulation_step = accumulation_step
        self.lr = self.optimizer.param_groups[0]['lr']
        
        self.scaler = GradScaler()
        self.optimizer.zero_grad()


    def train(self, dataloader, epoch):
        """
        default train
        """
        epoch_loss, step_loss = 0., 0.
        for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader),
                                         desc=" train", leave=False):
            if self.scheduler is not None:
                self.lr = self.scheduler.get_last_lr()[0]
            inputs = inputs.to(self.device)
            if isinstance(targets, torch.Tensor):
                targets = targets.to(self.device)

            with autocast():
                preds = self.model(inputs)
                batch_loss = self.criterion(preds, targets)
            
            # Accumulates scaled gradients.
            self.scaler.scale(batch_loss).backward()
            epoch_loss += batch_loss.detach()
            step_loss += batch_loss.detach()

            if (i+1) % self.accumulation_step == 0:
                self.scaler.step(self.optimizer)
                if self.scheduler:
                    type(self).apply_scheduler_step(self.scheduler,
                                                    epoch = epoch,
                                                    step = i,
                                                    steps_per_epoch = len(dataloader),
                                                    accumulation_step = self.accumulation_step
                                                    )
                self.scaler.update()
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
