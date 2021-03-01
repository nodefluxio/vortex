import torch
import torch.nn as nn

from .utils.focal_loss import FocalLoss


class ClassificationLoss(nn.Module):
    def __init__(self, loss: str = 'ce_loss', *args, **kwargs):
        super(ClassificationLoss, self).__init__()
        self.loss = loss
        if self.loss == 'ce_loss':
            self.loss_fn = nn.CrossEntropyLoss(*args, **kwargs)
        elif self.loss == 'focal_loss':
            self.loss_fn = FocalLoss(*args, **kwargs)
        else:
            raise RuntimeError("Unknown classifcation loss name, should be either 'ce_loss' or "
                "'focal_loss', got {}".format(loss))

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 2:
            target = target.squeeze(1)
        return self.loss_fn(input, target)
