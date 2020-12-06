import logging
import warnings

import torch
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)

class StepLRWithWarmUp(_LRScheduler):

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warm_up: int,
                 steps: list,
                 scales: list,
                 last_epoch=-1):
        """Step LR scheduler with burn in (warm up)
        
        Arguments:
            optimizer {torch.optim.Optimizer} -- Optimizer
            warm_up {int} -- Epochs for warm up
            steps {list} -- List of epoch when the learning rate will be reduced, e.g. [180,190] --> learning rate will be reduced on epoch 180 and epoch 190
            scales {list} -- Scale of the reduced learning rate, e.g. [0.1,0.1] --> e.g. initial lr == 0.01 , on epoch 180 will be reduced to 0.1 * 0.01 = 0.001 and on epoch 190 will be reduced to 0.1 * 0.001 = 0.0001
        
        Keyword Arguments:
            last_epoch {int} -- last epoch number (default: {-1})
        """

        assert warm_up > 0
        assert len(steps) == len(scales)
        for step in steps:
            assert step > warm_up
        for scale in scales:
            assert scale > 0

        self.warm_up = warm_up
        self.steps = steps
        self.scales = scales
        self.last_epoch = last_epoch

        super(StepLRWithWarmUp, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", DeprecationWarning)

        if self.last_epoch < self.warm_up:
            return [base_lr * pow((self.last_epoch + 1) / self.warm_up, 4) for base_lr in self.base_lrs]
        elif self.last_epoch < self.steps[0]:
            return [base_lr for base_lr in self.base_lrs]
        else:
            gamma = 1
            lrs = []
            for step, scale in zip(self.steps, self.scales):
                gamma *= scale
                if self.last_epoch >= step:
                    lrs = [base_lr * gamma for base_lr in self.base_lrs]
            return lrs
