from vortex.networks.modules import backbones
import torch
import pytest

no_pretrained = ['darknet53', 'shufflenetv2_x1.5', 'shufflenetv2_x2.0']
no_pretrained.extend(backbones.mobilenetv3.supported_models)
no_pretrained.remove("mobilenetv3_large_w1")
all_backbone = [m.__name__.split('.')[-1] for m in list(backbones.supported_models.keys())]


@pytest.mark.parametrize(
    "module, feature",
    [(bb, f) for bb in all_backbone for f in backbones.supported_feature_type]
)
def test_backbone(module, feature):
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


@pytest.mark.parametrize(
    "module, feature",
    [(bb, f) for bb in all_backbone for f in backbones.supported_feature_type]
)
def test_backbone_fail(module, feature):
    avail_model = getattr(backbones, module).supported_models
    for name in avail_model:
        print(name, feature)
        if name not in no_pretrained and feature == "classifier":
            network = backbones.get_backbone(name, pretrained=True, feature_type=feature, n_classes=2)
        else:
            network = backbones.get_backbone(name, pretrained=False, feature_type=feature, n_classes=2)

        channel = 2
        with pytest.raises(RuntimeError) as e :
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
