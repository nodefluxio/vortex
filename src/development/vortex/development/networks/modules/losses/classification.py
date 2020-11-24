import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.focal_loss import FocalLoss

class ClassificationLoss(nn.Module):
    def __init__(self, loss : str = 'ce_loss',*args, **kwargs):
        super(ClassificationLoss, self).__init__()
        self.loss=loss
        if self.loss == 'ce_loss':
            self.loss_fn = nn.NLLLoss(*args, **kwargs)
        elif self.loss == 'focal_loss':
            self.loss_fn = FocalLoss(*args, **kwargs)
        else:
            raise RuntimeError("Unknown classifcation loss, should be either 'ce_loss' or 'focal_loss', got {}".format(loss))

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.squeeze()
        if target.size() == torch.Size([]):
            target = target.unsqueeze(0)
        # x = self.loss_fn(input.log(), target)
        if self.loss == 'ce_loss':
            x = self.loss_fn(F.log_softmax(input,dim=1), target)
        else:
            x = self.loss_fn(input, target)
        return x
