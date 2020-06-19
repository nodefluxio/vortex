import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

from vortex.networks.modules import backbones
import torch

no_pretrained = ['darknet53', 'shufflenetv2_x1.5', 'shufflenetv2_x2.0']
no_pretrained.extend(backbones.mobilenetv3.supported_models)
no_pretrained.remove("mobilenetv3_large_w1")

def _common_test(module, feature):
    avail_model = getattr(backbones, module).supported_models
    for name in avail_model:
        print(name, feature)
        if name not in no_pretrained and feature == "classifier":
            network = backbones.get_backbone(name, pretrained=True, feature_type=feature, n_classes=2)
        else:
            network = backbones.get_backbone(name, pretrained=False, feature_type=feature, n_classes=2)

        x = torch.rand(1, 3, 224, 224)
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


def test_darknet53():
    _common_test("darknet53", "tri_stage_fpn")

def test_darknet53_classifier():
    _common_test("darknet53", "classifier")

def test_efficientnet():
    _common_test("efficientnet", "tri_stage_fpn")

def test_efficientnet_classifier():
    _common_test("efficientnet", "classifier")

def test_mobilenetv2():
    _common_test("mobilenetv2", "tri_stage_fpn")

def test_mobilenetv2_classifier():
    _common_test("mobilenetv2", "classifier")

def test_mobilenetv3():
    _common_test("mobilenetv3", "tri_stage_fpn")

def test_mobilenetv3_classifier():
    _common_test("mobilenetv3", "classifier")

def test_resnet():
    _common_test("resnet", "tri_stage_fpn")

def test_resnet_classifier():
    _common_test("resnet", "classifier")

def test_shufflenetv2():
    _common_test("shufflenetv2", "tri_stage_fpn")

def test_shufflenetv2_classifier():
    _common_test("shufflenetv2", "classifier")

def test_vgg():
    _common_test("vgg", "tri_stage_fpn")

def test_vgg_classifier():
    _common_test("vgg", "classifier")


if __name__ == "__main__":
    test_darknet53()
    test_darknet53_classifier()
    test_efficientnet()
    test_efficientnet_classifier()
    test_mobilenetv2()
    test_mobilenetv2_classifier()
    test_mobilenetv3()
    test_mobilenetv3_classifier()
    test_resnet()
    test_resnet_classifier()
    test_shufflenetv2()
    test_shufflenetv2_classifier()
    test_vgg()
    test_vgg_classifier()
