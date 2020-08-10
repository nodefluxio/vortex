import os
import sys
from pathlib import Path
proj_path = str(Path(__file__).parents[1])
sys.path.append(proj_path)
sys.path.append('vortex/runtime_package')
sys.path.append('vortex/development_package')


import torch
import pytest

from vortex.development.networks.models import create_model_components
from vortex.development.networks.modules.backbones import supported_models as supported_backbone
from vortex.development.utils.parser.parser import load_config, check_config

backbones = [bb[0] for bb in supported_backbone.values() if not 'mobilenetv3' in bb[0]]
backbones.append("mobilenetv3_large_w1")
backbones.remove('alexnet') if 'alexnet' in backbones else None
backbones.remove('squeezenetv1.0') if 'squeezenetv1.0' in backbones else None
backbones.remove('squeezenetv1.1') if 'squeezenetv1.1' in backbones else None
tasks = ["detection", "classification"]

@pytest.mark.parametrize(
    "task, backbone",
    [(t, bb) for t in tasks for bb in backbones]
)
def test_model(task, backbone):
    config_path = os.path.join(proj_path, "tests", "config", "test_" + task + ".yml")
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

    if task == 'classification':
        t = torch.randint(0, num_classes, (1,))
        assert x.size() == torch.Size([1, num_classes]), \
            "expected output size of %s for backbone '%s', got %s" % \
            (torch.Size([1, num_classes]), backbones[0], x.size())
    elif task == 'detection':
        t = torch.tensor([[0.0000, 14.0000,  0.4604,  0.0915,  0.2292,  0.3620],
                        [0.0000, 12.0000,  0.0896,  0.1165,  0.7583,  0.6617],
                        [0.0000, 14.0000,  0.1958,  0.2705,  0.0729,  0.0978]])
        assert len(x) == 3, "expected output to have 3 elements, got %s" % len(x)
        assert x[0].size(-1) == num_classes+5, "expected output model elements to have "\
            "torch.Size([*, %s]), got %s" % (num_classes+5, x[0].size())
    assert model.network.task == task
    l = model.loss(x, t)


if __name__ == "__main__":
    test_model(tasks[0], backbones[1])
