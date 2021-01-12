import pytest
import torch
import torch.nn as nn

from vortex.development.utils.common import create_optimizer, create_scheduler
from vortex.development.utils import lr_scheduler


@pytest.mark.parametrize(
    "method",
    [
        'SGD',
        'Adam',
        pytest.param('Invalid', marks=pytest.mark.xfail)
    ]
)
def test_create_optimizer(method):
    config = dict(
        trainer=dict(
            optimizer=dict(
                method=method,
                args=dict(lr=0.01)
            )
        )
    )
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3),
        nn.Linear(16, 5)
    )

    optimizer = create_optimizer(config, model.parameters())
    assert isinstance(optimizer, getattr(torch.optim, method))


@pytest.mark.parametrize(
    "with_monitor",
    [False, True],
    ids=["without monitor", "with monitor"]
)
def test_create_scheduler(with_monitor):
    config = dict(
        trainer=dict(
            optimizer=dict(
                method='SGD',
                args=dict(lr=0.01)
            )
        )
    )
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3),
        nn.Linear(16, 5)
    )
    optimizer = create_optimizer(config, model.parameters())

    modules = ['StepLR', 'CosineLRWithWarmUp']
    args = [dict(step_size=4), dict(t_initial=2)]
    for module, arg in zip(modules, args):
        config['trainer'].update(dict(lr_scheduler=dict(method=module, args=arg)))
        if with_monitor:
            config['trainer']['lr_scheduler'].update(dict(monitor='val_loss'))

        cfg = create_scheduler(config, optimizer)
        assert isinstance(cfg['lr_scheduler'], getattr(lr_scheduler, module))
        if with_monitor:
            assert 'monitor' in cfg and cfg['monitor'] == 'val_loss'
        else:
            assert not 'monitor' in cfg

    ## invalid method
    with pytest.raises(RuntimeError):
        config['trainer'].update(lr_scheduler=dict(method='Invalid', args=dict()))
        cfg = create_scheduler(config, optimizer)
