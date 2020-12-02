import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1].joinpath('src', 'development')))

from vortex.development.networks.modules import backbones
from vortex.development.networks.modules.utils.layers import EvoNormBatch2d
from vortex.development.networks.modules.utils import inplace_abn

import torch
import pytest

no_pretrained = ['darknet21', 'shufflenetv2_x1.5', 'shufflenetv2_x2.0']
all_backbone = [m.__name__.split('.')[-1] for m in list(backbones.supported_models.keys())]
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
    "module, feature",
    [(bb, f) for bb in all_backbone for f in backbones.supported_feature_type]
)
def test_backbone(module, feature):
    avail_model = getattr(backbones, module).supported_models
    for name in avail_model:
        if name in exclude_test:
            continue
        print(name, feature)
        pretrained = False
        if name not in no_pretrained and feature == "classifier":
            pretrained = True
        elif name != 'darknet53' and feature == 'tri_stage_fpn':
            network = backbones.get_backbone(name, pretrained=pretrained, feature_type=feature, 
                n_classes=2, norm_layer=EvoNormBatch2d, norm_kwargs={'eps': 1e-3})
            assert all(not isinstance(m, torch.nn.BatchNorm2d) for m in network.modules())
        network = backbones.get_backbone(name, pretrained=pretrained, feature_type=feature, n_classes=2)

        x = torch.rand(2, 3, 224, 224)
        x = network(x)

        if feature == "tri_stage_fpn":
            out_run = [o.size(1) for o in x]
            out_def = list(network.out_channels)[2:]
            assert out_run == out_def, "channel size defined in '{}' model ({}) " \
                "is not equal to output in runtime ({}).".format(name, out_def, out_run)
        elif feature == "classifier":
            assert x.size(1) == network.out_channels, "channel size defined in '{}' model ({}) is " \
                "not equal to output in runtime ({}).".format(name, network.out_channels, x.size())
        del network, x


@pytest.mark.parametrize(
    "module, feature",
    [(bb, f) for bb in all_backbone for f in backbones.supported_feature_type]
)
def test_backbone_fail(module, feature):
    avail_model = getattr(backbones, module).supported_models
    for name in avail_model:
        if name in exclude_test:
            continue
        print(name, feature)
        pretrained = False
        if name not in no_pretrained and feature == "classifier":
            pretrained = True
        network = backbones.get_backbone(name, pretrained=pretrained, feature_type=feature, n_classes=2)

        channel = 2
        with pytest.raises(RuntimeError):
            x = torch.rand(1, channel, 224, 224)
            x = network(x)

            if feature == "tri_stage_fpn":
                out_run = [o.size(1) for o in x]
                out_def = list(network.out_channels)[2:]
                assert out_run != out_def, "channel size defined in '{}' model ({}) " \
                    "is not equal to output in runtime ({}).".format(name, out_def, out_run)
            elif feature == "classifier":
                assert x.size(1) != network.out_channels, "channel size defined in '{}' model ({}) is " \
                    "not equal to output in runtime ({}).".format(name, network.out_channels, x.size())
            del network, x
