from pathlib import Path

import torch
import pytest

from torch import optim
from easydict import EasyDict
from copy import deepcopy

from ..common import (
    patched_pl_trainer, prepare_model,
    MINIMAL_TRAINER_CFG,
    DummyPLDatset
)

# from vortex.development.networks.models import create_model_components
# from vortex.development.networks.modules.backbones import supported_models as supported_backbone
# from vortex.development.utils.parser.parser import load_config, check_config
# from vortex.development.networks.modules.utils import inplace_abn

# proj_path = Path(__file__).parents[2]

# backbones = [bb[0] for bb in supported_backbone.values()]
# backbones.insert(0, 'darknet53')
# skip_backbone = [
#     'alexnet', 'squeezenetv1.0', 'squeezenetv1.1', 
#     'rexnet_100',   ## rexnet can't be tested for 1 batch training 
#     'resnest14',    ## resnest can't be tested for 1 batch training
#     'darknet7'      ## unusual spatial size for stage 4
# ]

# for b in skip_backbone:
#     if b in backbones:
#         backbones.remove(b)
# if not inplace_abn.has_iabn:    ## tresnet required additional module to be installed (inplace_abn)
#     backbones.remove('tresnet_m')

# tasks = ["detection", "classification"]


# @pytest.mark.parametrize(
#     "task, backbone",
#     [(t, bb) for t in tasks for bb in backbones]
# )
# def test_model(task, backbone):
#     config_path = proj_path.joinpath("tests", "config", "test_{}.yml".format(task))
#     config = load_config(config_path)
#     check_result = check_config(config, experiment_type='train')
#     assert check_result.valid, "config file %s for task %s is not valid, "\
#         "result:\n%s" % (config_path, task, str(check_result))

#     config.model.network_args.backbone = backbone
#     args = {
#         'model_name': config.model.name,
#         'preprocess_args': config.model.preprocess_args,
#         'network_args': config.model.network_args,
#         'loss_args': config.model.loss_args,
#         'postprocess_args': config.model.postprocess_args,
#         'stage': 'train'
#     }
#     model = create_model_components(**args)
#     num_classes = config.model.network_args.n_classes
#     assert hasattr(model.network, "output_format"), "model {} doesn't have 'output_format' "\
#         "attribute explaining the output of the model".format(config.model.name)

#     x = torch.randn(1, 3, 640, 640)
#     x = model.network(x)

#     t = torch.tensor(0)
#     if task == 'classification':
#         t = torch.randint(0, num_classes, (1,))
#         assert x.size() == torch.Size([1, num_classes]), \
#             "expected output size of %s for backbone '%s', got %s" % \
#             (torch.Size([1, num_classes]), backbones[0], x.size())
#     elif task == 'detection':
#         t = torch.tensor([[[14.0000,  0.4604,  0.0915,  0.2292,  0.3620],
#                            [12.0000,  0.0896,  0.1165,  0.7583,  0.6617],
#                            [14.0000,  0.1958,  0.2705,  0.0729,  0.0978]]])
#         assert len(x) == 3, "expected output to have 3 elements, got %s" % len(x)
#         assert x[0].size(-1) == num_classes+5, "expected output model elements to have "\
#             "torch.Size([*, %s]), got %s" % (num_classes+5, x[0].size())
#     assert model.network.task == task
#     l = model.loss(x, t)


@pytest.mark.parametrize(
    ("optim_method", "scheduler", "gpus"),
    [
        pytest.param("SGD", None, None, id="sgd scheduler None"),
        pytest.param("SGD", False, None, id="sgd no scheduler"),
        pytest.param("SGD", True, None, id="sgd with scheduler"),
        pytest.param(
            "SGD", True, 1, id="sgd in gpu", 
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU")
        ),
        pytest.param("Adam", True, None, id="adam with scheduler")
    ]
)
def test_configure_optimizer(tmp_path, optim_method, scheduler, gpus):
    config = EasyDict(deepcopy(MINIMAL_TRAINER_CFG))
    config['trainer']['optimizer'] = dict(
        method=optim_method,
        args=dict(lr=0.1)
    )
    if scheduler is True:
        scheduler = dict(
            method="StepLR",
            args=dict(step_size=1)
        )
    if scheduler is not False: ## if None or True
        config['trainer']['lr_scheduler'] = scheduler

    model = prepare_model(config)
    ## model.configure_optimizers called inside trainer.init_optimizers
    ## which also validates the returned value
    trainer_args = dict(
        logger=False, checkpoint_callback=False,
        limit_train_batches=2, limit_val_batches=2,
        progress_bar_refresh_rate=0, max_epochs=2,
        num_sanity_val_steps=0, weights_summary=None,
    )
    trainer = patched_pl_trainer(str(tmp_path), model, gpus=gpus, trainer_args=trainer_args)
    assert len(trainer.optimizers) == 1
    assert isinstance(trainer.optimizers[0], getattr(optim, optim_method))
    assert len(trainer.lr_schedulers) == (1 if scheduler else 0)
    if scheduler:
        assert isinstance(trainer.lr_schedulers[0]['scheduler'], optim.lr_scheduler.StepLR)
        assert trainer.lr_schedulers[0]['scheduler'].step_size == 1
        trainer.lr_schedulers[0]['interval'] == "epoch"

    datamodule = DummyPLDatset()
    trainer.fit(model, datamodule)


@pytest.mark.parametrize(
    "scheduler",
    [
        pytest.param(True, id="with scheduler"),
        pytest.param(False, id="without scheduler")
    ]
)
def test_model_get_lr(tmp_path, scheduler):
    scheduler = True
    learning_rate = 0.1
    config = EasyDict(deepcopy(MINIMAL_TRAINER_CFG))
    config['trainer']['optimizer'] = dict(
        method="SGD",
        args=dict(lr=learning_rate)
    )
    if scheduler:
        config['trainer']['lr_scheduler'] = dict(
            method="StepLR",
            args=dict(step_size=1)
        )

    model = prepare_model(config)
    trainer = patched_pl_trainer(str(tmp_path), model)
    model.trainer = trainer

    ret_lr = model.get_lr()
    assert ret_lr == learning_rate
