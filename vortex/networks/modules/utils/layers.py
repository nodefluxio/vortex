import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """function drop paths (Stochastic Depth) per sample.

    This is the same as the drop_connect implementation in original EfficientNet model.
    The original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper.
    """
    assert drop_prob >= 0
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    random_tensor = keep_prob + torch.rand((x.size()[0], 1, 1, 1), dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.ModuleDict):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    This is the same as the DropConnect implementation in original EfficientNet model.
    The original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper.
    """
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
