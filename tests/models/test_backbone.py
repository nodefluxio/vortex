from vortex.development.networks.modules import backbones
from vortex.development.networks.modules.utils.layers import EvoNormBatch2d
from vortex.development.networks.modules.utils import inplace_abn

import torch
import torch.nn as nn
import pytest

from copy import deepcopy


no_pretrained = [
    'darknet21', 'shufflenetv2_x1_5', 'shufflenetv2_x2_0',
    'rexnetr_100', 'rexnetr_130', 'rexnetr_150', 'rexnetr_200'
]
all_backbone = [m.__name__.split('.')[-1] for m in backbones.supported_models]
exclude_test = [ ## exclude bigger models
    'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 
    'efficientnet_b8', 'efficientnet_l2', 'efficientnet_l2_475', 
    'vgg13', 'vgg16_bn', 'vgg19_bn', 'vgg19',
    'resnet101', 'resnet152', 'resnext101_32x8d', 'wide_resnet101_2',
    'rexnet_200', 'resnest200', 'resnest269', 'resnest50d_1s4x24d',
    *backbones.regnet.supported_models[6:],
]
if not inplace_abn.has_iabn:
    exclude_test.extend(backbones.tresnet.supported_models)
else:
    exclude_test.extend(backbones.tresnet.supported_models[3:])

@pytest.mark.parametrize(
    "module",
    [bb for bb in all_backbone]
)
def test_backbone(module):
    default_cfgs = getattr(backbones, module).default_cfgs
    for name, cfg in default_cfgs.items():
        if name in exclude_test:
            continue
        print(name)
        pretrained = False
        if name not in no_pretrained:
            pretrained = True

        network = backbones.get_backbone(name, pretrained=False, n_classes=2,
            norm_layer=EvoNormBatch2d, norm_kwargs={'eps': 1e-3})
        assert all(not isinstance(m, torch.nn.BatchNorm2d) for m in network.modules())
        assert isinstance(network, backbones.BackboneBase)
        assert network.default_config == cfg
        assert network.name == name

        network = backbones.get_backbone(name, pretrained=pretrained, n_classes=2)

        stages = network.get_stages()
        assert len(stages) == 5
        assert isinstance(stages, (nn.Sequential, nn.ModuleList))

        classifier = network.get_classifier()
        assert isinstance(classifier, nn.Module)

        x = torch.rand(2, 3, 224, 224)
        out_run = []
        with torch.no_grad():
            for stage in stages:
                x = stage(x)
                out_run.append(x.size(1))
            x = classifier(x)

        out_def = network.stages_channel
        assert isinstance(out_def, tuple)
        assert tuple(out_run) == out_def, "channel size defined in '{}' model ({}) " \
            "is not equal to output in runtime ({}).".format(name, out_def, out_run)

        assert x.size(1) == network.num_classes, "channel size defined in '{}' model ({}) is " \
            "not equal to output in runtime ({}).".format(name, network.num_classes, x.size())
        del network, x


@pytest.mark.parametrize(
    "module",
    [bb for bb in all_backbone]
)
def test_backbone_fail(module):
    avail_model = getattr(backbones, module).supported_models
    for name in avail_model:
        if name in exclude_test:
            continue
        print(name)
        pretrained = False
        if name not in no_pretrained:
            pretrained = True

        network = backbones.get_backbone(name, pretrained=pretrained, n_classes=2)
        assert isinstance(network, backbones.BackboneBase)

        channel = 2
        with pytest.raises(RuntimeError):
            x = torch.rand(1, channel, 224, 224)
            x = network(x)


@pytest.mark.parametrize(
    "module",
    [bb for bb in all_backbone if bb != 'tresnet']
)
def test_reset_classifier(module):
    avail_model = getattr(backbones, module).supported_models
    backbone = backbones.get_backbone(avail_model[0])
    assert isinstance(backbone, backbones.BackboneBase)
    backbone.requires_grad_(False)
    backbone = backbone.eval()

    num_feature = backbone.num_classifer_feature
    num_classes = 9
    x = torch.randn(1, 3, 224, 224)

    ## change num_classes with default layer
    bb1 = deepcopy(backbone)
    bb1.reset_classifier(num_classes)
    bb1_out = bb1(x.clone())
    assert bb1.num_classes == num_classes
    assert bb1_out.size(1) == num_classes

    ## custom layer
    classifier = nn.Sequential(
        nn.Linear(num_feature, 512),
        nn.Dropout2d(0.2),
        nn.Linear(512, num_classes)
    )
    bb2 = deepcopy(backbone)
    bb2.reset_classifier(num_classes, classifier)
    bb2_out = bb2(x.clone())
    assert bb1.num_classes == num_classes
    assert bb2_out.size(1) == num_classes

    ## num_classes is negative
    bb3 = deepcopy(backbone)
    bb3.reset_classifier(-1)
    bb3_out = bb3(x.clone())
    assert bb3_out.size(1) == bb3.num_classifer_feature

    ## unknown type
    with pytest.raises(TypeError):
        backbone.reset_classifier(8, 'dummy')

