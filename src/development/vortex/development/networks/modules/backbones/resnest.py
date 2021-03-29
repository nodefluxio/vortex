""" ResNeSt Models
Paper: `ResNeSt: Split-Attention Networks` - https://arxiv.org/abs/2004.08955
Adapted from original PyTorch impl w/ weights at https://github.com/zhanghang1989/ResNeSt by Hang Zhang

Pretrained model from Ross Wightman
Modified by Vortex Team
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.arch_utils import load_pretrained
from .base_backbone import BackboneConfig, BackboneBase

_complete_url = lambda x: 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/' + x
_gluon_url = lambda x: 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/' + x
default_cfgs = {
    'resnest14': BackboneConfig(pretrained_url=_gluon_url('gluon_resnest14-9c8fe254.pth')),
    'resnest26': BackboneConfig(pretrained_url=_gluon_url('gluon_resnest26-50eb607c.pth')),
    'resnest50': BackboneConfig(pretrained_url=_complete_url('resnest50-528c19ca.pth')),
    'resnest101': BackboneConfig(
        pretrained_url=_complete_url('resnest101-22405ba7.pth'), input_size=(3, 256, 256)
    ),
    'resnest200': BackboneConfig(
        pretrained_url=_complete_url('resnest200-75117900.pth'), input_size=(3, 320, 320)
    ),
    'resnest269': BackboneConfig(
        pretrained_url=_complete_url('resnest269-0cc87c48.pth'), input_size=(3, 416, 416)
    ),
    'resnest50d_4s2x40d': BackboneConfig(pretrained_url=_complete_url('resnest50_fast_4s2x40d-41d14ed0.pth')),
    'resnest50d_1s4x24d': BackboneConfig(pretrained_url=_complete_url('resnest50_fast_1s4x24d-d4a4f76f.pth'))
}
supported_models = list(default_cfgs.keys())


class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        if self.radix > 1:
            batch = x.size(0)
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitAttConv2d(nn.Module):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True, radix=2, reduction_factor=4,
                 norm_layer=None, norm_kwargs=None, dropblock_prob=0.0, **kwargs):
        super(SplitAttConv2d, self).__init__()
        padding = nn.modules.utils._pair(padding)
        norm_kwargs = norm_kwargs or {}
        assert dropblock_prob == 0.0, "dropblock is not yet supported"

        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels

        self.conv = nn.Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                              groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels*radix, **norm_kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels, **norm_kwargs)
        self.fc2 = nn.Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        splited = []
        if self.radix > 1:
            if torch.__version__ < '1.5':
                splited = torch.split(x, int(rchannel//self.radix), dim=1)
            else:
                splited = torch.split(x, rchannel//self.radix, dim=1)
            gap = sum(splited) 
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            if torch.__version__ < '1.5':
                attens = torch.split(atten, int(rchannel//self.radix), dim=1)
            else:
                attens = torch.split(atten, rchannel//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()


class ResNestBottleneck(nn.Module):
    """ResNest Bottleneck
    """

    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, radix=1, cardinality=1, 
                 bottleneck_width=64, avd=False, avd_first=False, dilation=1, is_first=False,
                 dropblock_prob=0.0, zero_init_last_bn=False, norm_layer=None, norm_kwargs=None):
        super(ResNestBottleneck, self).__init__()
        assert dropblock_prob == 0.0, "dropblock is not yet supported"
        norm_layer = norm_layer or nn.BatchNorm2d
        norm_kwargs = norm_kwargs or {}

        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if radix >= 1:
            self.conv2 = SplitAttConv2d(group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation, dilation=dilation, groups=cardinality, 
                bias=False, radix=radix, norm_layer=norm_layer, dropblock_prob=dropblock_prob)
        else:
            self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation, groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv2d(group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes*4)

        if zero_init_last_bn:
            nn.init.zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 0:
            out = self.bn2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNest(BackboneBase):
    """ResNest Variants

    NOTE: ResNest variant can't be trained with single batch data using BatchNorm2d,
        use different batch normalization layer if you need to, e.g. InstanceNorm2d.
    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the 
            IEEE conference on computer vision and pattern recognition. 2016.
        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    def __init__(self, name, block, layers, num_classes=1000, radix=1, cardinality=1, bottleneck_width=64,
                 dilated=False, dilation=1, deep_stem=False, stem_width=64, avg_down=False,
                 avd=False, avd_first=False, final_drop=0.0, dropblock_prob=0, 
                 zero_init_last_bn=False, norm_layer=nn.BatchNorm2d, norm_kwargs=None, default_config=None):
        norm_kwargs = norm_kwargs or {}
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width*2 if deep_stem else 64
        self.stem_channel = stem_width*2 if deep_stem else 64
        self.avg_down = avg_down
        self.zero_init_last_bn = zero_init_last_bn
        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first

        self.block_expansion = block.expansion

        super(ResNest, self).__init__(name, default_config)
        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(stem_width, **norm_kwargs),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(stem_width, **norm_kwargs),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes, **norm_kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, 
                norm_layer=norm_layer, norm_kwargs=norm_kwargs, dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, 
                norm_layer=norm_layer, norm_kwargs=norm_kwargs, dropblock_prob=dropblock_prob)
        elif dilation==2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=1, 
                norm_layer=norm_layer, norm_kwargs=norm_kwargs, dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, 
                norm_layer=norm_layer, norm_kwargs=norm_kwargs, dropblock_prob=dropblock_prob)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                norm_layer=norm_layer, norm_kwargs=norm_kwargs, dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                norm_layer=norm_layer, norm_kwargs=norm_kwargs, dropblock_prob=dropblock_prob)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.fc = nn.Linear(512 * self.block_expansion, num_classes)

        stages_channel = [self.stem_channel] + [x * self.block_expansion for x in [64, 128, 256, 512]]
        self._stages_channel = tuple(stages_channel)
        self._num_classes = num_classes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None,
                    dropblock_prob=0.0, is_first=True, norm_kwargs={}):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride,
                                                    ceil_mode=True, count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1,
                                                    ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=stride, bias=False))
            down_layers.append(norm_layer(planes * block.expansion, **norm_kwargs))
            downsample = nn.Sequential(*down_layers)

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width, avd=self.avd, 
                                avd_first=self.avd_first, dilation=1, 
                                is_first=is_first, dropblock_prob=dropblock_prob,
                                norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                zero_init_last_bn=self.zero_init_last_bn))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width, avd=self.avd, 
                                avd_first=self.avd_first, dilation=2, is_first=is_first,
                                norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                dropblock_prob=dropblock_prob,
                                zero_init_last_bn=self.zero_init_last_bn))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, radix=self.radix, 
                                cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=dilation, dropblock_prob=dropblock_prob,
                                norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                zero_init_last_bn=self.zero_init_last_bn))

        return nn.Sequential(*layers)

    def get_stages(self):
        return nn.Sequential(
            nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool),
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4
        )

    @property
    def stages_channel(self):
        return self._stages_channel

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_classifer_feature(self):
        return 512*self.block_expansion

    def get_classifier(self):
        return nn.Sequential(
            self.avgpool,
            nn.Flatten(start_dim=1),
            self.fc
        )

    def reset_classifier(self, num_classes, classifier=None):
        self._num_classes = num_classes
        if num_classes < 0:
            classifier = nn.Identity()
        elif classifier is None:
            classifier = nn.Linear(self.num_classifer_feature, num_classes)
        if not isinstance(classifier, nn.Module):
            raise TypeError("'classifier' argument is required to have type of 'int' or 'nn.Module', "
                "got {}".format(type(classifier)))
        self.fc = classifier

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.drop:
            x = self.drop(x)
        x = self.fc(x)

        return x


def _resnest(arch, layers, pretrained, progress, **kwargs):
    num_classes = 1000
    if pretrained and kwargs.get("num_classes", False):
        num_classes = kwargs.pop("num_classes")

    model = ResNest(arch, ResNestBottleneck, layers, default_config=default_cfgs[arch], **kwargs)
    if pretrained:
        load_pretrained(model, default_cfgs[arch].pretrained_url, num_classes=num_classes, 
            first_conv_name="conv1", classifier_name="fc", progress=progress)
    return model


def resnest14(pretrained=False, progress=True, **kwargs):
    """ ResNeSt-14 model Weights ported from GluonCV.
    """
    return _resnest('resnest14', [1, 1, 1, 1], pretrained, progress, deep_stem=True, 
        stem_width=32, avg_down=True, bottleneck_width=64, cardinality=1, radix=2,
        avd=True, avd_first=False, **kwargs)

def resnest26(pretrained=False, progress=True, **kwargs):
    """ ResNeSt-126 model Weights ported from GluonCV.
    """
    return _resnest('resnest26', [2, 2, 2, 2], pretrained, progress, deep_stem=True, 
        stem_width=32, avg_down=True, bottleneck_width=64, cardinality=1, radix=2,
        avd=True, avd_first=False, **kwargs)

def resnest50(pretrained=False, progress=True, **kwargs):
    """ ResNeSt-50 model from https://arxiv.org/abs/2004.08955
    """
    return _resnest('resnest50', [3, 4, 6, 3], pretrained, progress, deep_stem=True, 
        stem_width=32, avg_down=True, bottleneck_width=64, cardinality=1, radix=2,
        avd=True, avd_first=False, **kwargs)

def resnest101(pretrained=False, progress=True, **kwargs):
    """ ResNeSt-101 model from https://arxiv.org/abs/2004.08955
    """
    return _resnest('resnest101', [3, 4, 23, 3], pretrained, progress, deep_stem=True, 
        stem_width=64, avg_down=True, bottleneck_width=64, cardinality=1, radix=2,
        avd=True, avd_first=False, **kwargs)

def resnest200(pretrained=False, progress=True, **kwargs):
    """ ResNeSt-200 model from https://arxiv.org/abs/2004.08955
    """
    return _resnest('resnest200', [3, 24, 36, 3], pretrained, progress, deep_stem=True, 
        stem_width=64, avg_down=True, bottleneck_width=64, cardinality=1, radix=2,
        avd=True, avd_first=False, **kwargs)

def resnest269(pretrained=False, progress=True, **kwargs):
    """ ResNeSt-269 model from https://arxiv.org/abs/2004.08955
    """
    return _resnest('resnest269', [3, 30, 48, 8], pretrained, progress, deep_stem=True, 
        stem_width=64, avg_down=True, bottleneck_width=64, cardinality=1, radix=2,
        avd=True, avd_first=False, **kwargs)

def resnest50d_4s2x40d(pretrained=False, progress=True, **kwargs):
    """ ResNeSt-50 4s2x40d model from https://github.com/zhanghang1989/ResNeSt/blob/master/ablation.md
    """
    return _resnest('resnest50d_4s2x40d', [3, 4, 6, 3], pretrained, progress, deep_stem=True, 
        stem_width=32, avg_down=True, bottleneck_width=40, cardinality=2, radix=4,
        avd=True, avd_first=True, **kwargs)

def resnest50d_1s4x24d(pretrained=False, progress=True, **kwargs):
    """ ResNeSt-50 1s4x24d from https://github.com/zhanghang1989/ResNeSt/blob/master/ablation.md
    """
    return _resnest('resnest50d_1s4x24d', [3, 4, 6, 3], pretrained, progress, deep_stem=True, 
        stem_width=32, avg_down=True, bottleneck_width=24, cardinality=4, radix=1,
        avd=True, avd_first=True, **kwargs)
