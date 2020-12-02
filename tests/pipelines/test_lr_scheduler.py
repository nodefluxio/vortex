import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2].joinpath('src', 'development')))

import math
import torch
import unittest
import torchvision

from vortex.development.core.engine.trainer.lr_scheduler import CosineLRWithWarmUp

class TestLRScheduler(unittest.TestCase) :
    def test_cosine(self) :
        alexnet = torchvision.models.alexnet()
        optimizer = torch.optim.SGD(
            alexnet.parameters(),
            lr=1e-2, momentum=0.9
        )
        num_epochs = 100
        lr_min = 1e-5
        warmup_lr = 1e-3
        warmup_epoch = 3
        scheduler = CosineLRWithWarmUp(
            optimizer,
            t_initial=num_epochs,
            t_mul=1.0,
            lr_min=lr_min,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_epoch,
            cycle_limit=1,
            t_in_epochs=False
        )
        num_epochs = scheduler.get_cycle_length()
        lr = []
        for i in range(num_epochs+warmup_epoch) :
            lr.append(scheduler.get_update_values(i))
        self.assertTrue(math.isclose(lr[0][0],warmup_lr))
        self.assertTrue(math.isclose(lr[-1][0],lr_min))