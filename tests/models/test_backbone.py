from vortex.development.networks.modules import backbones
from vortex.development.networks.modules.utils.layers import EvoNormBatch2d
from vortex.development.networks.modules.utils import inplace_abn
from vortex.development.networks.models.backbone import Backbone

import torch
import torch.nn as nn
import numpy as np
import onnxruntime
import pytest

from copy import deepcopy
from torch.onnx import export as onnx_export
from torch.jit import trace as torchscript_trace, load as torchscript_load


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
        assert all(not isinstance(m, nn.BatchNorm2d) for m in network.modules())
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


def test_backbone_class_validate_stages():
    num_stages = 6

    ## str stages output
    stages_out_c = Backbone._validate_stages_output("classifier", num_stages, Backbone._stages_output_str_map)
    stages_out_ts = Backbone._validate_stages_output("tri_stage", num_stages, Backbone._stages_output_str_map)
    stages_out_tsf = Backbone._validate_stages_output("tri_stage_fpn", num_stages, Backbone._stages_output_str_map)
    assert stages_out_c == (num_stages-1,)
    assert stages_out_ts == (2, 3, 4)
    assert stages_out_ts == stages_out_tsf

    ## str stages not available
    with pytest.raises(ValueError):
        Backbone._validate_stages_output("invalid", num_stages, Backbone._stages_output_str_map)

    ## int stages
    stages_out = Backbone._validate_stages_output(4, num_stages, Backbone._stages_output_str_map)
    assert stages_out == (4,)
    stages_out = Backbone._validate_stages_output(-2, num_stages, Backbone._stages_output_str_map)
    assert stages_out == (4,)

    ## list stages
    stages_out = Backbone._validate_stages_output([2, 4], num_stages, Backbone._stages_output_str_map)
    assert stages_out == (2, 4)
    stages_out = Backbone._validate_stages_output((2, 4), num_stages, Backbone._stages_output_str_map)
    assert stages_out == (2, 4)

    ## stages higher than num_stages
    with pytest.raises(RuntimeError):
        stages_out = Backbone._validate_stages_output([2, 4, 6], num_stages, Backbone._stages_output_str_map)

    with pytest.raises(RuntimeError):
        stages_out = Backbone._validate_stages_output(6, num_stages, Backbone._stages_output_str_map)

    with pytest.raises(RuntimeError): ## too negative
        stages_out = Backbone._validate_stages_output(-7, num_stages, Backbone._stages_output_str_map)

    ## invalid stages output
    with pytest.raises(TypeError):
        stages_out = Backbone._validate_stages_output({"invalid": [2, 3, 4]}, num_stages, Backbone._stages_output_str_map)


def test_backbone_class():
    x = torch.randn(2, 3, 224, 224)

    ## normal
    model = Backbone("resnet18", stages_output="tri_stage", norm_layer=EvoNormBatch2d)
    with torch.no_grad():
        y = model(x)
    assert model.name == "resnet18"
    assert model.stages_output == (2, 3, 4)
    assert model.stages_channel == (128, 256, 512)
    assert len(model.hooks_handle) == len(model.stages_output)
    assert len(model) == 5
    assert all(not isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert tuple(o.size(1) for o in y) == model.stages_channel

    model = Backbone("resnet18", stages_output="classifier")
    with torch.no_grad():
        y = model(x)
    assert model.name == "resnet18"
    assert model.stages_output == (5,)
    assert model.stages_channel == (1000,)
    assert len(model.hooks_handle) == len(model.stages_output)
    assert len(model) == 6    ## 5 stages with classifier
    assert tuple(o.size(1) for o in y) == model.stages_channel

    module = backbones.get_backbone("resnet18")
    model = Backbone(module, stages_output="tri_stage")
    assert model.name == "resnet18"
    assert model.stages_output == (2, 3, 4)
    assert model.stages_channel == (128, 256, 512)
    model = Backbone(module, stages_output="classifier")
    assert model.name == "resnet18"
    assert model.stages_output == (5,)
    assert model.stages_channel == (1000,)


    stages_channel = [128, 256, 512]
    num_classes = 10
    stages = [
        nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        ),
        nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        ),
        nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        ),
    ]
    classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(1),
        nn.Linear(512, num_classes)
    )

    ## defined module
    model = Backbone(stages, stages_output=[0,1,2], stages_channel=stages_channel, name="dummy")
    assert model.name == "dummy"
    assert model.stages_output == (0,1,2)
    assert model.stages_channel == tuple(stages_channel)

    model = Backbone(nn.Sequential(*stages), stages_output=[0,1,2], stages_channel=stages_channel, name="dummy")
    assert model.name == "dummy"
    assert model.stages_output == (0,1,2)
    assert model.stages_channel == tuple(stages_channel)

    with pytest.raises(RuntimeError):
        Backbone(stages, stages_output=[0,1], stages_channel=stages_channel, name="dummy")

    ## infer channel output
    model = Backbone(stages, stages_output=[0,1,2])
    assert model.stages_output == (0,1,2)
    assert model.stages_channel == tuple(stages_channel)

    ## defined module with classifier
    module = stages + [classifier]
    model = Backbone(module, stages_output="classifier", stages_channel=[num_classes], name="dummy")
    assert model.name == "dummy"
    assert model.stages_output == (3,)
    assert model.stages_channel == (num_classes,)

    model = Backbone(nn.Sequential(*module), stages_output="classifier", stages_channel=[num_classes], name="dummy")
    assert model.name == "dummy"
    assert model.stages_output == (3,)
    assert model.stages_channel == (num_classes,)

    ## infer channel output
    model = Backbone(module, stages_output=-1)
    assert model.stages_output == (3,)
    assert model.stages_channel == (num_classes,)

    model = Backbone("resnet18", stages_output="classifier", freeze=True)
    assert all(not p.requires_grad for p in model.parameters())

    ## invalid module
    with pytest.raises(TypeError):
        Backbone({"invalid": stages}, stages_output=[0,1,2], stages_channel=stages_channel)


def test_backbone_class_export(tmp_path):
    stages_output = "tri_stage"

    x = torch.randn(1, 3, 224, 224)
    model = Backbone("resnet18", stages_output=stages_output, freeze=True).eval()
    y = model(x)
    assert tuple(o.size(1) for o in y) == model.stages_channel

    ## onnx
    onnx_path = tmp_path.joinpath("resnet18_backbone.onnx")
    onnx_export(model, x, str(onnx_path), input_names=['input'], opset_version=11)
    sess = onnxruntime.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    ort_y = sess.run(None, {input_name: x.clone().numpy()})
    assert tuple(o.shape[1] for o in ort_y) == model.stages_channel
    assert all([np.allclose(r, p.numpy(), atol=1e-4) for r,p in zip(ort_y, y)])

    pt_path = onnx_path.with_suffix(".pt")
    pt_model = torchscript_trace(model, x, check_tolerance=1e-6)
    pt_model.save(str(pt_path))
    loaded_pt_model = torchscript_load(str(pt_path))
    pt_y = loaded_pt_model(x.clone())
    assert tuple(o.size(1) for o in pt_y) == model.stages_channel
    assert all([torch.allclose(p, pr) for p,pr in zip(y, pt_y)])
