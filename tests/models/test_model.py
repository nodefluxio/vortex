import sys
from pathlib import Path
proj_path = Path(__file__).parents[2]
sys.path.insert(0, str(proj_path.joinpath('src', 'runtime')))
sys.path.insert(0, str(proj_path.joinpath('src', 'development')))

import torch
import pytest

from vortex.development.networks.models import create_model_components
from vortex.development.networks.modules.backbones import supported_models as supported_backbone
from vortex.development.utils.parser.parser import load_config, check_config
from vortex.development.networks.modules.utils import inplace_abn

backbones = [bb[0] for bb in supported_backbone.values()]
backbones.insert(0, 'darknet53')
skip_backbone = [
    'alexnet', 'squeezenetv1.0', 'squeezenetv1.1', 
    'rexnet_100',   ## rexnet can't be tested for 1 batch training 
    'resnest14',    ## resnest can't be tested for 1 batch training
    'darknet7'      ## unusual spatial size for stage 4
]

for b in skip_backbone:
    if b in backbones:
        backbones.remove(b)
if not inplace_abn.has_iabn:    ## tresnet required additional module to be installed (inplace_abn)
    backbones.remove('tresnet_m')

tasks = ["detection", "classification"]


@pytest.mark.parametrize(
    "task, backbone",
    [(t, bb) for t in tasks for bb in backbones]
)
def test_model(task, backbone):
    config_path = proj_path.joinpath("tests", "config", "test_{}.yml".format(task))
    config = load_config(config_path)
    check_result = check_config(config, experiment_type='train')
    assert check_result.valid, "config file %s for task %s is not valid, "\
        "result:\n%s" % (config_path, task, str(check_result))

    config.model.network_args.backbone = backbone
    if backbones[0] == 'darknet53':
        config.model.network_args.pretrained_backbone = None
    args = {
        'model_name': config.model.name,
        'preprocess_args': config.model.preprocess_args,
        'network_args': config.model.network_args,
        'loss_args': config.model.loss_args,
        'postprocess_args': config.model.postprocess_args,
        'stage': 'train'
    }
    model = create_model_components(**args)
    num_classes = config.model.network_args.n_classes
    assert hasattr(model.network, "output_format"), "model {} doesn't have 'output_format' "\
        "attribute explaining the output of the model".format(config.model.name)

    x = torch.randn(1, 3, 640, 640)
    x = model.network(x)

    t = torch.tensor(0)
    if task == 'classification':
        t = torch.randint(0, num_classes, (1,))
        assert x.size() == torch.Size([1, num_classes]), \
            "expected output size of %s for backbone '%s', got %s" % \
            (torch.Size([1, num_classes]), backbones[0], x.size())
    elif task == 'detection':
        t = torch.tensor([[[14.0000,  0.4604,  0.0915,  0.2292,  0.3620],
                           [12.0000,  0.0896,  0.1165,  0.7583,  0.6617],
                           [14.0000,  0.1958,  0.2705,  0.0729,  0.0978]]])
        assert len(x) == 3, "expected output to have 3 elements, got %s" % len(x)
        assert x[0].size(-1) == num_classes+5, "expected output model elements to have "\
            "torch.Size([*, %s]), got %s" % (num_classes+5, x[0].size())
    assert model.network.task == task
    l = model.loss(x, t)
