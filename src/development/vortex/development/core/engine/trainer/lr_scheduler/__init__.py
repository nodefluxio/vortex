from torch.optim.lr_scheduler import *
from .cosine_lr_with_warmup import *
from .tanh_lr_with_warmup import *
from .step_lr_with_warmup import *

step_update_map = {
    'batch_update':['CyclicLR',
                    'OneCycleLR',
                    ],
    'epoch_update':['StepLRWithWarmUp',
                    'CosineLRWithWarmUp',
                    'TanhLRWithWarmUp',
                    'StepLR',
                    'MultiStepLR',
                    'ExponentialLR',
                    'CosineAnnealingLR',
                    'CosineAnnealingWarmRestarts'
                    ]
}