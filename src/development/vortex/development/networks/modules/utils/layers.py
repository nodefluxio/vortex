""" Layers compilation

Hacked together by / Copyright 2020 Ross Wightman
Edited by Vortex Team
"""

import torch
import torch.nn as nn

from .conv2d import create_conv2d
from .activations import sigmoid, get_act_layer
from .arch_utils import make_divisible

TF_BN_MOMENTUM = 1 - 0.99
TF_BN_EPSILON = 1e-3

def resolve_se_args(kwargs, in_channel, act_layer=None):
    se_kwargs = {
        'reduce_mid': False,
        'divisor': 1,
        'act_layer': None,
        'gate_fn': sigmoid
    }
    if kwargs is not None:
        se_kwargs.update(kwargs)
    if not se_kwargs.get('reduce_mid', False):
        se_kwargs['reduced_base_chs'] = in_channel
    if not 'act_layer' in se_kwargs or se_kwargs['act_layer'] is None:
        se_kwargs['act_layer'] = act_layer
    return se_kwargs

def resolve_act_layer(kwargs, default='relu'):
    act_layer = kwargs.pop('act_layer', default)
    if isinstance(act_layer, str):
        act_layer = get_act_layer(act_layer)
    return act_layer

def resolve_norm_layer(kwargs, default=nn.BatchNorm2d):
    norm_layer = kwargs.pop('norm_layer', default)
    if isinstance(norm_layer, nn.Module):
        raise RuntimeError("'norm_layer' arguments should be nn.Module instance")
    return norm_layer

def resolve_norm_args(kwargs):
    bn_args = {}
    if kwargs.pop('bn_tf', False):
        bn_args = dict(momentum=TF_BN_MOMENTUM, eps=TF_BN_EPSILON)
    bn_momentum = kwargs.pop('bn_momentum', None)
    if bn_momentum is not None:
        bn_args['momentum'] = bn_momentum
    bn_eps = kwargs.pop('bn_eps', None)
    if bn_eps is not None:
        bn_args['eps'] = bn_eps
    return bn_args


class SqueezeExcite(nn.Module):
    def __init__(self, in_channel, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=sigmoid, divisor=1, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = make_divisible((reduced_base_chs or in_channel) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_channel, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_channel, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x

class ConvBnAct(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, dilation=1, 
                 pad_type='', act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **_):
        super(ConvBnAct, self).__init__()
        norm_kwargs = norm_kwargs or {}
        self.conv = create_conv2d(in_channel, out_channel, kernel_size, stride=stride, 
            dilation=dilation, padding=pad_type)
        self.bn1 = norm_layer(out_channel, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class DepthwiseSeparableConv(nn.Module):
    """ DepthwiseSeparable block
    
    See Figure 7 on https://arxiv.org/abs/1807.11626
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). 
    This is an alternative to having a IR with an optional first pw conv.
    """
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, dilation=1, 
                 se_ratio=0., se_kwargs=None, pad_type='', act_layer=nn.ReLU, noskip=False, 
                 exp_ratio=1.0, drop_path_rate=0., norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(DepthwiseSeparableConv, self).__init__()

        assert kernel_size in [3, 5]
        norm_kwargs = norm_kwargs or {}
        has_se = se_ratio is not None and se_ratio > 0
        self.has_residual = (stride == 1 and in_channel == out_channel) and not noskip

        self.conv_dw = create_conv2d(in_channel, in_channel, kernel_size, stride=stride,
            dilation=dilation, padding=pad_type, depthwise=True, bias=False)
        self.bn1 = norm_layer(in_channel, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        # Squeeze-and-excitation
        if has_se:
            se_kwargs = resolve_se_args(se_kwargs, in_channel, act_layer)
            self.se = SqueezeExcite(in_channel, se_ratio=se_ratio, **se_kwargs)
        else:
            self.se = nn.Identity()

        self.conv_pw = create_conv2d(in_channel, out_channel, 1, padding=pad_type, bias=False)
        self.bn2 = norm_layer(out_channel, **norm_kwargs)
        if drop_path_rate > 0:
            self.drop_path = DropPath(drop_path_rate)
        else:
            self.drop_path = nn.Identity()

    def forward(self, x):
        residual = x

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.se(x)

        x = self.conv_pw(x)
        x = self.bn2(x)

        if self.has_residual:
            x = self.drop_path(x)
            x += residual
        return x


class InvertedResidualBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block
    
    See Figure 7 on https://arxiv.org/abs/1807.11626
    Based on MNASNet
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, dilation=1,
                 se_ratio=0., se_kwargs=None, pad_type='', act_layer=nn.ReLU, noskip=False, 
                 exp_ratio=1.0, drop_path_rate=0., norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(InvertedResidualBlock, self).__init__()

        assert kernel_size in [3, 5]
        norm_kwargs = norm_kwargs or {}
        has_se = se_ratio is not None and se_ratio > 0
        self.has_residual = (in_channel == out_channel and stride == 1) and not noskip

        ## Point-wise expansion -> _expand_conv in original implementation
        # 'conv_pw' could be by-passed when 'exp_ratio' is 1
        mid_chs = make_divisible(in_channel * exp_ratio)
        self.conv_pw = create_conv2d(in_channel, mid_chs, 1, padding=pad_type, bias=False)
        self.bn1 = norm_layer(mid_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        # Depth-wise convolution
        self.conv_dw = create_conv2d(mid_chs, mid_chs, kernel_size, stride=stride,
            dilation=dilation, padding=pad_type, bias=False, depthwise=True)
        self.bn2 = norm_layer(mid_chs, **norm_kwargs)
        self.act2 = act_layer(inplace=True)

        # Squeeze-and-excitation
        if has_se:
            se_kwargs = resolve_se_args(se_kwargs, in_channel, act_layer)
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio, **se_kwargs)
        else:
            self.se = nn.Identity()

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(mid_chs, out_channel, 1, padding=pad_type, bias=False)
        self.bn3 = norm_layer(out_channel, **norm_kwargs)
        if drop_path_rate > 0:
            self.drop_path = DropPath(drop_path_rate)
        else:
            self.drop_path = nn.Identity()

    def forward(self, x):
        residual = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Squeeze-and-excitation
        x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            x = self.drop_path(x)
            x += residual
        return x


class EdgeResidual(nn.Module):
    """ Residual block with expansion convolution followed by pointwise-linear w/ stride"""

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, dilation=1, 
                 se_ratio=0., se_kwargs=None, pad_type='', act_layer=nn.ReLU, noskip=False, 
                 exp_ratio=1.0, mid_channel=0, drop_path_rate=0., norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(EdgeResidual, self).__init__()

        assert kernel_size in [3, 5]
        norm_kwargs = norm_kwargs or {}
        has_se = se_ratio is not None and se_ratio > 0
        self.has_residual = (in_channel == out_channel and stride == 1) and not noskip

        # Expansion convolution
        if mid_channel > 0:
            mid_channel = make_divisible(mid_channel * exp_ratio)
        else:
            mid_channel = make_divisible(in_channel * exp_ratio)
        self.conv_exp = create_conv2d(in_channel, mid_channel, kernel_size, padding=pad_type, bias=False)
        self.bn1 = norm_layer(mid_channel, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        # Squeeze-and-excitation
        if has_se:
            se_kwargs = resolve_se_args(se_kwargs, in_channel, act_layer)
            self.se = SqueezeExcite(mid_channel, se_ratio=se_ratio, **se_kwargs)
        else:
            self.se = nn.Identity()

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(mid_channel, out_channel, 1, stride=stride, 
            dilation=dilation, padding=pad_type, bias=False)
        self.bn2 = norm_layer(out_channel, **norm_kwargs)
        if drop_path_rate > 0:
            self.drop_path = DropPath(drop_path_rate)
        else:
            self.drop_path = nn.Identity()

    def forward(self, x):
        residual = x

        # Expansion convolution
        x = self.conv_exp(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Squeeze-and-excitation
        x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn2(x)

        if self.has_residual:
            x = self.drop_path(x)
            x += residual
        return x


def drop_connect(inputs, training : bool=False, drop_connect_rate : float=0.):
    """Apply drop connect."""
    if not training:
        return inputs

    keep_prob = 1 - drop_connect_rate
    random_tensor = keep_prob + torch.rand(
        (inputs.size()[0], 1, 1, 1), dtype=inputs.dtype, device=inputs.device)
    random_tensor.floor_()  # binarize
    output = inputs.div(keep_prob) * random_tensor
    return output


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """function drop paths (Stochastic Depth) per sample.

    This is the same as the drop_connect implementation in original EfficientNet model.
    The original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper.
    """
    assert drop_prob >= 0
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    random_tensor = keep_prob + torch.rand((x.size()[0], 1, 1, 1), dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.ModuleDict):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    This is the same as the DropConnect implementation in original EfficientNet model.
    The original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper.
    """
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
