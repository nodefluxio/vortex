import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2].joinpath('src', 'development')))

from vortex.development.networks.modules.losses.utils.focal_loss import FocalLoss

import numpy as np
import torch

def test_focal_loss():
    test_input = np.random.rand(5,10)
    test_target = np.random.randint(10, size=(5,1))

    loss_fn = FocalLoss()
    loss = loss_fn(torch.tensor(test_input), torch.tensor(test_target))
    assert isinstance(loss, torch.Tensor)
