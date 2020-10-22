""" ReXNet

A PyTorch impl of `ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network` -
https://arxiv.org/abs/2007.00992

Adapted from original impl at https://github.com/clovaai/rexnet
Copyright (c) 2020-present NAVER Corp. MIT license

Changes for timm, feature extraction, and rounded channel variant hacked together by Ross Wightman
Copyright 2020 Ross Wightman

modified by Vortex Team
"""

import warnings
import torch.nn as nn
import math

from ..utils.arch_utils import make_divisible, load_pretrained
from ..utils.activations import get_act_layer
from ..utils.layers import ConvBnAct, ClassifierHead
from .base_backbone import Backbone, ClassifierFeature


_complete_url = lambda x: 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rexnet/' + x
model_urls = {
    'rexnet_100': _complete_url('rexnetv1_100-1b4dddf4.pth'),
    'rexnet_130': _complete_url('rexnetv1_130-590d768e.pth'),
    'rexnet_150': _complete_url('rexnetv1_150-bd1a6aa8.pth'),
    'rexnet_200': _complete_url('rexnetv1_200-8c0b7f2d.pth'),
}
supported_models = list(model_urls.keys())


class SEWithNorm(nn.Module):

    def __init__(self, channels, reduction=16, act_layer=nn.ReLU, divisor=1, reduction_channels=None,
                 gate_layer='sigmoid', norm_layer=None, norm_kwargs=None):
        super(SEWithNorm, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if norm_kwargs is None:
            norm_kwargs = {}

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduction_channels = reduction_channels or make_divisible(channels // reduction, divisor=divisor)
        self.fc1 = nn.Conv2d(channels, reduction_channels, kernel_size=1, padding=0, bias=True)
        self.bn = norm_layer(reduction_channels, **norm_kwargs)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(reduction_channels, channels, kernel_size=1, padding=0, bias=True)
        self.gate = get_act_layer(gate_layer)(inplace=False)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.fc1(x_se)
        x_se = self.bn(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)


class LinearBottleneck(nn.Module):
    def __init__(self, in_chs, out_chs, stride, exp_ratio=1.0, use_se=True, se_rd=12, ch_div=1, 
                 norm_layer=None, norm_kwargs=None):
        super(LinearBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if norm_kwargs is None:
            norm_kwargs = {}

        self.use_shortcut = stride == 1 and in_chs <= out_chs
        self.in_channels = in_chs
        self.out_channels = out_chs

        dw_chs = in_chs
        self.conv_exp = None
        if exp_ratio != 1.:
            dw_chs = make_divisible(round(in_chs * exp_ratio), divisor=ch_div)
            self.conv_exp = ConvBnAct(in_chs, dw_chs, act_layer=get_act_layer("swish"), 
                norm_layer=norm_layer, norm_kwargs=norm_kwargs)

        self.conv_dw = ConvBnAct(dw_chs, dw_chs, 3, stride=stride, groups=dw_chs, apply_act=False,
            norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        self.se = None
        if use_se:
            self.se = SEWithNorm(dw_chs, reduction=se_rd, divisor=ch_div, norm_layer=norm_layer, 
                norm_kwargs=norm_kwargs)
        self.act_dw = nn.ReLU6()

        self.conv_pwl = ConvBnAct(dw_chs, out_chs, 1, apply_act=False, norm_layer=norm_layer, 
            norm_kwargs=norm_kwargs)

    def forward(self, x):
        shortcut = x
        if self.conv_exp is not None:
            x = self.conv_exp(x)
        x = self.conv_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.act_dw(x)
        x = self.conv_pwl(x)
        if self.use_shortcut:
            x[:, 0:self.in_channels] += shortcut
        return x

class ReXNetV1(nn.Module):
    """ReXNet model variant

    NOTE: ReXNet variant can't be trained with single batch data using BatchNorm2d,
        use different batch normalization layer if you need to, e.g. InstanceNorm2d.
    """
    def __init__(self, in_channel=3, num_classes=1000, output_stride=32, norm_layer=None, norm_kwargs=None,
                 initial_chs=16, final_chs=180, width_mult=1.0, depth_mult=1.0, use_se=True,
                 se_rd=12, ch_div=1, drop_rate=0.2):
        super(ReXNetV1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if norm_kwargs is None:
            norm_kwargs = {}
        self.drop_rate = drop_rate

        assert output_stride == 32  # FIXME support dilation
        stem_base_chs = 32 / width_mult if width_mult < 1.0 else 32
        stem_chs = make_divisible(round(stem_base_chs * width_mult), divisor=ch_div)
        self.stem = ConvBnAct(in_channel, stem_chs, 3, stride=2, act_layer=get_act_layer('swish'),
            norm_layer=norm_layer, norm_kwargs=norm_kwargs)

        self.out_channels = [stem_chs]

        block_cfg = self._block_cfg(width_mult, depth_mult, initial_chs, final_chs, use_se, ch_div)
        features = self._build_blocks(block_cfg, stem_chs, width_mult, norm_layer, se_rd=se_rd, 
            ch_div=ch_div, norm_kwargs=norm_kwargs)
        self.num_features = self.out_channels[-1]
        self.features = nn.Sequential(*features)

        self.head = ClassifierHead(self.num_features, num_classes, drop_rate=drop_rate)
        self.out_channels.append(num_classes)

        # FIXME weight init, the original appears to use PyTorch defaults

    @staticmethod
    def _block_cfg(width_mult=1.0, depth_mult=1.0, initial_chs=16, final_chs=180, use_se=True, ch_div=1):
        layers = [1, 2, 2, 3, 3, 5]
        strides = [1, 2, 2, 2, 1, 2]
        layers = [math.ceil(element * depth_mult) for element in layers]
        strides = sum([[element] + [1] * (layers[idx] - 1) for idx, element in enumerate(strides)], [])
        exp_ratios = [1] * layers[0] + [6] * sum(layers[1:])
        depth = sum(layers[:]) * 3
        base_chs = initial_chs / width_mult if width_mult < 1.0 else initial_chs

        # The following channel configuration is a simple instance to make each layer become an expand layer.
        out_chs_list = []
        for i in range(depth // 3):
            out_chs_list.append(make_divisible(round(base_chs * width_mult), divisor=ch_div))
            base_chs += final_chs / (depth // 3 * 1.0)

        if use_se:
            use_ses = [False] * (layers[0] + layers[1]) + [True] * sum(layers[2:])
        else:
            use_ses = [False] * sum(layers[:])
        return zip(out_chs_list, exp_ratios, strides, use_ses)

    def _build_blocks(self, block_cfg, prev_chs, width_mult, norm_layer, se_rd=12, ch_div=1, norm_kwargs={}):
        features = []
        for chs, exp_ratio, stride, se in block_cfg:
            if stride > 1:
                self.out_channels.append(features[-1].out_channels)
            features.append(LinearBottleneck(in_chs=prev_chs, out_chs=chs, exp_ratio=exp_ratio, stride=stride, 
                use_se=se, se_rd=se_rd, ch_div=ch_div, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            prev_chs = chs
        self.out_channels.append(features[-1].out_channels)
        pen_chs = make_divisible(1280 * width_mult, divisor=ch_div)
        features.append(ConvBnAct(prev_chs, pen_chs, act_layer=get_act_layer("swish"), 
            norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        self.out_channels.append(pen_chs)
        return features

    def get_stages(self):
        out_channels = self.out_channels[2:-1].copy()
        stages = [
            nn.Sequential(self.stem, *self.features[0:3]),
            self.features[3:5],
            self.features[5:11],
            self.features[11:-1],
            self.features[-1]
        ]
        return nn.Sequential(*stages), out_channels

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.head = ClassifierHead(self.num_features, num_classes, drop_rate=self.drop_rate)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.features(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _rexnet(arch, pretrained, progress, **kwargs):
    num_classes = 1000
    if pretrained and "num_classes" in kwargs:
        num_classes = kwargs.pop("num_classes")

    model = ReXNetV1(**kwargs)
    if pretrained and arch in model_urls:
        load_pretrained(model, model_urls[arch], num_classes=num_classes, 
            first_conv_name="stem.conv", classifier_name="head.fc", progress=progress)
    else:
        if pretrained:
            warnings.warn("backbone {} does not have pretrained model.".format(arch))
        if num_classes != 1000:
            model.reset_classifier(num_classes)
    return model


def rexnet_100(pretrained=False, progress=True, **kwargs):
    """ReXNet V1 1.0x"""
    return _rexnet('rexnet_100', pretrained, progress, **kwargs)


def rexnet_130(pretrained=False, progress=True, **kwargs):
    """ReXNet V1 1.3x"""
    return _rexnet('rexnet_130', pretrained, progress, width_mult=1.3, **kwargs)


def rexnet_150(pretrained=False, progress=True, **kwargs):
    """ReXNet V1 1.5x"""
    return _rexnet('rexnet_150', pretrained, progress, width_mult=1.5, **kwargs)


def rexnet_200(pretrained=False, progress=True, **kwargs):
    """ReXNet V1 2.0x"""
    return _rexnet('rexnet_200', pretrained, progress, width_mult=2.0, **kwargs)


def rexnetr_100(pretrained=False, progress=True, **kwargs):
    """ReXNet V1 1.0x w/ rounded (mod 8) channels"""
    return _rexnet('rexnetr_100', pretrained, progress, ch_div=8, **kwargs)


def rexnetr_130(pretrained=False, progress=True, **kwargs):
    """ReXNet V1 1.3x w/ rounded (mod 8) channels"""
    return _rexnet('rexnetr_130', pretrained, progress, width_mult=1.3, ch_div=8, **kwargs)


def rexnetr_150(pretrained=False, progress=True, **kwargs):
    """ReXNet V1 1.5x w/ rounded (mod 8) channels"""
    return _rexnet('rexnetr_150', pretrained, progress, width_mult=1.5, ch_div=8, **kwargs)


def rexnetr_200(pretrained=False, progress=True, **kwargs):
    """ReXNet V1 2.0x w/ rounded (mod 8) channels"""
    return _rexnet('rexnetr_200', pretrained, progress, width_mult=2.0, ch_div=8, **kwargs)


def get_backbone(model_name : str, pretrained: bool = False, feature_type: str = "tri_stage_fpn", 
                 n_classes: int = 1000, **kwargs):
    if not model_name in supported_models:
        raise RuntimeError("model {} is not supported, available: {}".format(model_name, supported_models))

    network = eval('{}(pretrained=pretrained, num_classes=n_classes, **kwargs)'.format(model_name))
    stages, channels = network.get_stages()

    if feature_type == "tri_stage_fpn":
        backbone = Backbone(stages, channels)
    elif feature_type == "classifier":
        backbone = ClassifierFeature(stages, network.get_classifier(), n_classes)
    else:
        raise NotImplementedError("'feature_type' for other than 'tri_stage_fpn' and 'classifier'"\
            "is not currently implemented, got %s" % (feature_type))
    return backbone
