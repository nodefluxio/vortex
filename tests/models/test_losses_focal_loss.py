from vortex.development.networks.modules.losses.utils.focal_loss import FocalLoss

import torch

def test_focal_loss():
    test_input = torch.rand(5,10)
    test_target = torch.randint(10, size=(5, 1))

    loss_fn = FocalLoss()
    loss = loss_fn(test_input, test_target)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() ## scalar
