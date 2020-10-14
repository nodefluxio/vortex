import math
import torch
import torch.nn as nn

from collections import OrderedDict
from .base_backbone import Backbone, ClassifierFeature
from ..utils.darknet import load_darknet_weight
from ..utils.arch_utils import load_pretrained


_dropbox_url = lambda sh, fname: "https://www.dropbox.com/s/{}/{}?dl=1".format(sh, fname)

model_urls = {
    ## note that darknet7 model only has backbone weight (without classifier)
    'darknet7': _dropbox_url("dhu17cc41fi4kyz", "darknet7_bb-8ef63008.pth"), 
    'darknet19': _dropbox_url("0dr0fk2dmjmm0ld", "darknet19-814e7f8a.pth"),
    'darknet21': None,
    'darknet53': _dropbox_url("1k7oa4edm1jyd9t", "darknet53-81b1ed5f.pth")
}

supported_models = list(model_urls.keys())


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, norm_kwargs=None):
        super(BasicBlock, self).__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        norm_kwargs = norm_kwargs or {}

        kernel_size = 3
        if inplanes > outplanes:
            kernel_size = 1
        padding = (kernel_size-1) // 2
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, 
                              stride=1, padding=padding, bias=False)
        self.bn = norm_layer(outplanes)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Bottleneck(nn.Module):
    """ Building block for darknet21 and darknet53
    """
    def __init__(self, planes, outplanes, norm_layer=None, norm_kwargs=None):
        super(Bottleneck, self).__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        norm_kwargs = norm_kwargs or {}

        self.conv1 = nn.Conv2d(outplanes, planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(planes, outplanes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(outplanes)
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out


class DarkNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, in_channel=3, 
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(DarkNet, self).__init__()
        norm_kwargs = norm_kwargs or {}
        self.inplanes = 32
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channel, self.inplanes, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes, **norm_kwargs)
        self.relu1 = nn.LeakyReLU(0.1)

        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer, norm_kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], norm_layer, norm_kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], norm_layer, norm_kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], norm_layer, norm_kwargs)
        self.layer5 = self._make_layer(block, 1024, layers[4], norm_layer, norm_kwargs)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

        self.out_channels = [self.inplanes, 64, 128, 256, 512, 1024, num_classes]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, norm_layer, norm_kwargs):
        layers = []
        if block == BasicBlock:
            assert blocks % 2 == 1
            #  downsample
            layers.append(("ds_pool", nn.MaxPool2d(kernel_size=2, stride=2)))

            #  blocks
            inplanes = self.inplanes
            outplanes = planes
            for i in range(blocks):
                block_layer = BasicBlock(inplanes, outplanes, norm_layer, norm_kwargs)
                layers.append((f"block_{i}", block_layer))
                inplanes, outplanes = outplanes, inplanes
            assert inplanes == planes
        elif block == Bottleneck:
            #  downsample
            layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes, kernel_size=3,
                                    stride=2, padding=1, bias=False)))
            layers.append(("ds_bn", norm_layer(planes, **norm_kwargs)))
            layers.append(("ds_relu", nn.LeakyReLU(0.1)))

            #  blocks
            for i in range(blocks):
                block_layer = Bottleneck(self.inplanes, planes, norm_layer, norm_kwargs)
                layers.append(("residual_{}".format(i), block_layer))
        self.inplanes = planes
        return nn.Sequential(OrderedDict(layers))

    def get_stages(self):
        out_channels = self.out_channels[1:-1].copy()
        stages = [
            nn.Sequential(self.conv1, self.bn1, self.relu1, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        ]
        return nn.Sequential(*stages), out_channels

    def get_classifier(self):
        return nn.Sequential(
            self.avgpool,
            nn.Flatten(start_dim=1),
            self.fc
        )

    def reset_classifier(self, num_classes):
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


def _darknet(arch, block, layers, pretrained, progress, **kwargs):
    num_classes = 1000
    if pretrained and kwargs.get("num_classes", False):
        num_classes = kwargs.pop("num_classes")

    model = DarkNet(block, layers, **kwargs)
    if pretrained:
        if model_urls[arch] is not None:
            load_pretrained(model, model_urls[arch], num_classes=num_classes, 
                first_conv_name="conv1", classifier_name="fc", progress=progress)
        elif isinstance(pretrained, str):
            if pretrained.endswith('.pth'):
                load_pretrained(model, pretrained, num_classes=num_classes, 
                    first_conv_name="conv1", classifier_name="fc", progress=progress)
            elif pretrained.endswith('.weights') or pretrained.split('.')[0] == arch:
                load_darknet_weight(pretrained)
            else:
                raise RuntimeError("Unknown pretrained model weight of {}".format(pretrained))
        else:
            raise RuntimeError("Pretrained model is not available")
    return model


def darknet7(pretrained=False, progress=True, **kwargs):
    r""" Construct a darknet-7 model used in 
    `"Tiny-YOLOv3" <https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg>`_ and 
    `"Tiny-YOLOv2" <https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-tiny.cfg>`_
    """
    return _darknet("darknet7", BasicBlock, [1, 1, 1, 1, 1], pretrained, progress, **kwargs)

def darknet19(pretrained=False, progress=True, **kwargs):
    r"""Constructs a darknet-19 model from 
    `"YOLO9000: Better, Faster, Stronger" <http://arxiv.org/abs/1612.08242>`_
    """
    return _darknet("darknet19", BasicBlock, [1, 3, 3, 5, 5], pretrained, progress, **kwargs)


def darknet21(pretrained=False, progress=True, **kwargs):
    """Constructs a darknet-21 model.
    """
    return _darknet("darknet21", Bottleneck, [1, 1, 2, 2, 1], pretrained, progress, **kwargs)


def darknet53(pretrained=False, progress=True, **kwargs):
    """Constructs a darknet-53 model from
    `"YOLOv3: An Incremental Improvement" <http://arxiv.org/abs/1804.02767>`_
    """
    return _darknet("darknet53", Bottleneck, [1, 2, 8, 8, 4], pretrained, progress, **kwargs)


def get_backbone(model_name : str, pretrained: bool = False, feature_type: str = "tri_stage_fpn", 
                 n_classes: int = 1000, *args, **kwargs):
    if not model_name in supported_models:
        raise RuntimeError("model %s is not supported, available: %s" %(model_name, supported_models))

    model = eval('{}(pretrained=pretrained, num_classes=n_classes, **kwargs)'.format(model_name))
    stages, channels = model.get_stages()

    if feature_type == "tri_stage_fpn":
        backbone = Backbone(stages, channels)
    elif feature_type == "classifier":
        backbone = ClassifierFeature(stages, model.get_classifier(), n_classes)
    else:
        raise NotImplementedError("'feature_type' for other than 'tri_stage_fpn' and 'classifier'"\
            "is not currently implemented, got %s" % (feature_type))
    return backbone
