"""
TResNet: High Performance GPU-Dedicated Architecture
https://arxiv.org/pdf/2003.13630.pdf

Original model: https://github.com/mrT23/TResNet

Hacked together by  Ross Wightman
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/tresnet.py

Modified by Vortex Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from collections import OrderedDict

from .base_backbone import BackboneConfig, BackboneBase
from ..utils.inplace_abn import InplaceAbn
from ..utils.layers import SEModule, ClassifierHead
from ..utils.arch_utils import load_pretrained


_complete_url = lambda x: 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/' + x
default_cfgs = {
    'tresnet_m': BackboneConfig(
        pretrained_url=_complete_url('tresnet_m_80_8-dbc13962.pth'),
        normalize_mean=(0, 0, 0), normalize_std=(1, 1, 1)
    ),
    'tresnet_l': BackboneConfig(
        pretrained_url=_complete_url('tresnet_l_81_5-235b486c.pth'),
        normalize_mean=(0, 0, 0), normalize_std=(1, 1, 1)
    ),
    'tresnet_xl': BackboneConfig(
        pretrained_url=_complete_url('tresnet_xl_82_0-a2d51b00.pth'),
        normalize_mean=(0, 0, 0), normalize_std=(1, 1, 1)
    ),
    'tresnet_m_448': BackboneConfig(
        pretrained_url=_complete_url('tresnet_m_448-bc359d10.pth'), input_size=(3, 448, 448),
        normalize_mean=(0, 0, 0), normalize_std=(1, 1, 1)
    ),
    'tresnet_l_448': BackboneConfig(
        pretrained_url=_complete_url('tresnet_l_448-940d0cd1.pth'), input_size=(3, 448, 448),
        normalize_mean=(0, 0, 0), normalize_std=(1, 1, 1)
    ),
    'tresnet_xl_448': BackboneConfig(
        pretrained_url=_complete_url('tresnet_xl_448-8c1815de.pth'), input_size=(3, 448, 448),
        normalize_mean=(0, 0, 0), normalize_std=(1, 1, 1)
    ),
}
supported_models = list(default_cfgs.keys())


def IABN2Float(module: nn.Module) -> nn.Module:
    """If `module` is IABN don't use half precision."""
    if isinstance(module, InplaceAbn):
        module.float()
    for child in module.children():
        IABN2Float(child)
    return module


def conv2d_iabn(ni, nf, stride, kernel_size=3, groups=1, act_layer="leaky_relu", act_param=1e-2):
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=kernel_size, stride=stride, 
            padding=kernel_size // 2, groups=groups, bias=False),
        InplaceAbn(nf, act_layer=act_layer, act_param=act_param)
    )


class SpaceToDepth(nn.Module):
    def __init__(self, block_size=4):
        super().__init__()
        assert block_size == 4
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x


@torch.jit.script
class SpaceToDepthJit(object):
    def __call__(self, x: torch.Tensor):
        # assuming hard-coded that block_size==4 for acceleration
        N, C, H, W = x.size()
        x = x.view(N, C, H // 4, 4, W // 4, 4)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * 16, H // 4, W // 4)  # (N, C*bs^2, H//bs, W//bs)
        return x


class SpaceToDepthModule(nn.Module):
    def __init__(self, no_jit=False):
        super().__init__()
        if not no_jit:
            self.op = SpaceToDepthJit()
        else:
            self.op = SpaceToDepth()

    def forward(self, x):
        return self.op(x)


class DepthToSpace(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x


class AntiAliasDownsampleLayer(nn.Module):
    def __init__(self, channels: int = 0, filt_size: int = 3, stride: int = 2, no_jit: bool = False):
        super(AntiAliasDownsampleLayer, self).__init__()
        if no_jit:
            self.op = Downsample(channels, filt_size, stride)
        else:
            self.op = DownsampleJIT(channels, filt_size, stride)

        # FIXME I should probably override _apply and clear DownsampleJIT filter cache for .cuda(), .half(), etc calls

    def forward(self, x):
        return self.op(x)


@torch.jit.script
class DownsampleJIT(object):
    def __init__(self, channels: int = 0, filt_size: int = 3, stride: int = 2):
        self.channels = channels
        self.stride = stride
        self.filt_size = filt_size
        assert self.filt_size == 3
        assert stride == 2
        self.filt = {}  # lazy init by device for DataParallel compat

    def _create_filter(self, like: torch.Tensor):
        filt = torch.tensor([1., 2., 1.], dtype=like.dtype, device=like.device)
        filt = filt[:, None] * filt[None, :]
        filt = filt / torch.sum(filt)
        return filt[None, None, :, :].repeat((self.channels, 1, 1, 1))

    def __call__(self, input: torch.Tensor):
        input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
        filt = self.filt.get(str(input.device), self._create_filter(input))
        return F.conv2d(input_pad, filt, stride=2, padding=0, groups=input.shape[1])


class Downsample(nn.Module):
    def __init__(self, channels=None, filt_size=3, stride=2):
        super(Downsample, self).__init__()
        self.channels = channels
        self.filt_size = filt_size
        self.stride = stride

        assert self.filt_size == 3
        filt = torch.tensor([1., 2., 1.])
        filt = filt[:, None] * filt[None, :]
        filt = filt / torch.sum(filt)

        # self.filt = filt[None, None, :, :].repeat((self.channels, 1, 1, 1))
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, input):
        input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
        return F.conv2d(input_pad, self.filt, stride=self.stride, padding=0, groups=input.shape[1])


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, aa_layer=None):
        super(BasicBlock, self).__init__()
        if stride == 1:
            self.conv1 = conv2d_iabn(inplanes, planes, stride=1, act_param=1e-3)
        else:
            if aa_layer is None:
                self.conv1 = conv2d_iabn(inplanes, planes, stride=2, act_param=1e-3)
            else:
                self.conv1 = nn.Sequential(
                    conv2d_iabn(inplanes, planes, stride=1, act_param=1e-3),
                    aa_layer(channels=planes, filt_size=3, stride=2))

        self.conv2 = conv2d_iabn(planes, planes, stride=1, act_layer="identity")
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        reduction_chs = max(planes * self.expansion // 4, 64)
        self.se = SEModule(planes * self.expansion, reduction_channels=reduction_chs) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.se is not None:
            out = self.se(out)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True,
                 act_layer="leaky_relu", aa_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv2d_iabn(
            inplanes, planes, kernel_size=1, stride=1, act_layer=act_layer, act_param=1e-3)
        if stride == 1:
            self.conv2 = conv2d_iabn(
                planes, planes, kernel_size=3, stride=1, act_layer=act_layer, act_param=1e-3)
        else:
            if aa_layer is None:
                self.conv2 = conv2d_iabn(
                    planes, planes, kernel_size=3, stride=2, act_layer=act_layer, act_param=1e-3)
            else:
                self.conv2 = nn.Sequential(
                    conv2d_iabn(planes, planes, kernel_size=3, stride=1, act_layer=act_layer, act_param=1e-3),
                    aa_layer(channels=planes, filt_size=3, stride=2))

        reduction_chs = max(planes * self.expansion // 8, 64)
        self.se = SEModule(planes, reduction_channels=reduction_chs) if use_se else None

        self.conv3 = conv2d_iabn(
            planes, planes * self.expansion, kernel_size=1, stride=1, act_layer="identity")

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None:
            out = self.se(out)

        out = self.conv3(out)
        out = out + residual  # no inplace
        out = self.relu(out)

        return out


class TResNet(BackboneBase):
    def __init__(self, name, layers, in_channel=3, num_classes=1000, width_factor=1.0, 
                 no_aa_jit=False, drop_rate=0., norm_layer=None, norm_kwargs=None,
                 default_config=None):
        super(TResNet, self).__init__(name, default_config)
        if norm_layer or norm_kwargs:
            import warnings
            warnings.warn("Can not change norm_layer for TResNet variant models, "
                "ignoring argument value.")

        self._num_classes = num_classes
        self.drop_rate = drop_rate

        # JIT layers
        space_to_depth = SpaceToDepthModule()
        aa_layer = partial(AntiAliasDownsampleLayer, no_jit=no_aa_jit)

        # TResnet stages
        self.inplanes = int(64 * width_factor)
        self.planes = int(64 * width_factor)
        conv1 = conv2d_iabn(in_channel * 16, self.planes, stride=1, kernel_size=3)
        layer1 = self._make_layer(BasicBlock, self.planes, 
            layers[0], stride=1, use_se=True, aa_layer=aa_layer)  # 56x56
        layer2 = self._make_layer(BasicBlock, self.planes * 2, 
            layers[1], stride=2, use_se=True, aa_layer=aa_layer)  # 28x28
        layer3 = self._make_layer(Bottleneck, self.planes * 4, 
            layers[2], stride=2, use_se=True, aa_layer=aa_layer)  # 14x14
        layer4 = self._make_layer(Bottleneck, self.planes * 8, 
            layers[3], stride=2, use_se=False, aa_layer=aa_layer)  # 7x7

        # body
        self.body = nn.Sequential(OrderedDict([
            ('SpaceToDepth', space_to_depth),
            ('conv1', conv1),
            ('layer1', layer1),
            ('layer2', layer2),
            ('layer3', layer3),
            ('layer4', layer4)
        ]))

        # head
        self.num_features = (self.planes * 8) * Bottleneck.expansion
        self.head = ClassifierHead(self.num_features, num_classes, pool_type='fast', drop_rate=drop_rate)

        self._stages_channel = (
            self.planes, 
            self.planes * BasicBlock.expansion, 
            self.planes * 2 * BasicBlock.expansion,
            self.planes * 4 * Bottleneck.expansion,
            self.planes * 8 * Bottleneck.expansion
        )

        # model initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, InplaceAbn):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # residual connections special initialization
        for m in self.modules():
            if isinstance(m, BasicBlock):
                m.conv2[1].weight = nn.Parameter(torch.zeros_like(m.conv2[1].weight))  # BN to zero
            if isinstance(m, Bottleneck):
                m.conv3[1].weight = nn.Parameter(torch.zeros_like(m.conv3[1].weight))  # BN to zero
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)

    def _make_layer(self, block, planes, blocks, stride=1, use_se=True, aa_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = []
            if stride == 2:
                # avg pooling before 1x1 conv
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False))
            layers += [conv2d_iabn(
                self.inplanes, planes * block.expansion, kernel_size=1, stride=1, act_layer="identity")]
            downsample = nn.Sequential(*layers)

        layers = []
        layers.append(block(
            self.inplanes, planes, stride, downsample, use_se=use_se, aa_layer=aa_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=use_se, aa_layer=aa_layer))
        return nn.Sequential(*layers)

    def get_stages(self):
        return nn.Sequential(
            nn.Sequential(self.body.SpaceToDepth, self.body.conv1),
            self.body.layer1,
            self.body.layer2,
            self.body.layer3,
            self.body.layer4,
        )

    @property
    def stages_channel(self):
        return self._stages_channel

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_classifer_feature(self):
        return self.num_features

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, classifier=None):
        self._num_classes = num_classes
        if num_classes < 0:
            classifier = nn.Identity()
        elif classifier is None:
            classifier = nn.Linear(self.num_features, num_classes)
        if not isinstance(classifier, nn.Module):
            raise TypeError("'classifier' argument is required to have type of 'int' or 'nn.Module', "
                "got {}".format(type(classifier)))
        self.head.fc = classifier

    def forward_features(self, x):
        return self.body(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def _tresnet(arch, layers, pretrained, progress, **kwargs):
    num_classes = 1000
    if pretrained and kwargs.get("num_classes", False):
        num_classes = kwargs.pop("num_classes")

    model = TResNet(arch, layers, default_config=default_cfgs[arch], **kwargs)
    if pretrained:
        load_pretrained(model, default_cfgs[arch].pretrained_url, num_classes=num_classes, 
            first_conv_name="body.conv1", classifier_name="head.fc", progress=progress)
    return model


def tresnet_m(pretrained=False, progress=True, **kwargs):
    return _tresnet('tresnet_m', pretrained=pretrained, progress=progress, 
        layers=[3, 4, 11, 3], **kwargs)

def tresnet_l(pretrained=False, progress=True, **kwargs):
    return _tresnet('tresnet_l', pretrained=pretrained, progress=progress, 
        layers=[4, 5, 18, 3], width_factor=1.2, **kwargs)

def tresnet_xl(pretrained=False, progress=True, **kwargs):
    return _tresnet('tresnet_xl', pretrained=pretrained, progress=progress, 
        layers=[4, 5, 24, 3], width_factor=1.3, **kwargs)

def tresnet_m_448(pretrained=False, progress=True, **kwargs):
    return _tresnet('tresnet_m_448', pretrained=pretrained, progress=progress,
        layers=[3, 4, 11, 3], **kwargs)

def tresnet_l_448(pretrained=False, progress=True, **kwargs):
    return _tresnet('tresnet_l_448', pretrained=pretrained, progress=progress, 
        layers=[4, 5, 18, 3], width_factor=1.2, **kwargs)

def tresnet_xl_448(pretrained=False, progress=True, **kwargs):
    return _tresnet('tresnet_xl_448', pretrained=pretrained, progress=progress, 
        layers=[4, 5, 24, 3], width_factor=1.3, **kwargs)
