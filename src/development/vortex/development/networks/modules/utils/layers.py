""" Layers compilation

Hacked together by / Copyright 2020 Ross Wightman
Edited by Vortex Team
"""

import torch
import torch.nn as nn

from .conv2d import create_conv2d
from .activations import sigmoid, get_act_layer
from .arch_utils import make_divisible

_SE_ARGS_DEFAULT = dict(
    gate_fn=sigmoid,
    act_layer=None,
    reduce_mid=False,
    divisor=1
)

def resolve_se_args(kwargs, in_chs, act_layer=None):
    se_kwargs = kwargs.copy() if kwargs is not None else {}
    # fill in args that aren't specified with the defaults
    for k, v in _SE_ARGS_DEFAULT.items():
        se_kwargs.setdefault(k, v)
    # some models, like MobilNetV3, calculate SE reduction chs from the containing block's mid_ch instead of in_ch
    if not se_kwargs.pop('reduce_mid'):
        se_kwargs['reduced_base_chs'] = in_chs
    # act_layer override, if it remains None, the containing block's act_layer will be used
    if se_kwargs['act_layer'] is None:
        assert act_layer is not None
        se_kwargs['act_layer'] = act_layer
    return se_kwargs


def resolve_act_layer(kwargs, default='relu'):
    act_layer = kwargs.pop('act_layer', default)
    if isinstance(act_layer, str):
        act_layer = get_act_layer(act_layer)
    return act_layer


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

class DepthwiseSeparableConv(nn.Module):
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """
    def __init__(self, in_chs, out_chs, dw_kernel_size=3,
                 stride=1, dilation=1, pad_type='', act_layer=nn.ReLU, noskip=False,
                 pw_kernel_size=1, pw_act=False, se_ratio=0., se_kwargs=None,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, drop_path_rate=0.):
        super(DepthwiseSeparableConv, self).__init__()
        norm_kwargs = norm_kwargs or {}
        has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act  # activation after point-wise conv
        self.drop_path_rate = drop_path_rate

        self.conv_dw = create_conv2d(
            in_chs, in_chs, dw_kernel_size, stride=stride, dilation=dilation, padding=pad_type, depthwise=True)
        self.bn1 = norm_layer(in_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        # Squeeze-and-excitation
        if has_se:
            se_kwargs = resolve_se_args(se_kwargs, in_chs, act_layer)
            self.se = SqueezeExcite(in_chs, se_ratio=se_ratio, **se_kwargs)
        else:
            self.se = torch.nn.Identity()

        self.conv_pw = create_conv2d(in_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_layer(out_chs, **norm_kwargs)
        self.act2 = act_layer(inplace=True) if self.has_pw_act else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PW
            info = dict(module='conv_pw', hook_type='forward_pre', num_chs=self.conv_pw.in_channels)
        else:  # location == 'bottleneck', block output
            info = dict(module='', hook_type='', num_chs=self.conv_pw.out_channels)
        return info

    def forward(self, x):
        residual = x

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.se(x)

        x = self.conv_pw(x)
        x = self.bn2(x)
        x = self.act2(x)

        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += residual
        return x


class InvertedResidualBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block
    
    See Figure 7 on https://arxiv.org/abs/1807.11626
    Based on MNASNet
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, se_ratio=0., 
                 pad_type='', act_layer=nn.ReLU, noskip=False, exp_ratio=1.0, 
                 drop_path_rate=0., norm_kwargs=None):
        super(InvertedResidualBlock, self).__init__()

        assert kernel_size in [3, 5]
        norm_kwargs = norm_kwargs or {}
        has_se = se_ratio is not None and se_ratio > 0
        self.has_residual = (in_channel == out_channel and stride == 1) and not noskip

        ## Point-wise expansion -> _expand_conv in original implementation
        # 'conv_pw' could be by-passed when 'exp_ratio' is 1
        mid_chs = make_divisible(in_channel * exp_ratio)
        self.conv_pw = create_conv2d(in_channel, mid_chs, 1, padding=pad_type, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        # Depth-wise convolution
        self.conv_dw = create_conv2d(mid_chs, mid_chs, kernel_size, stride=stride,
            padding=pad_type, bias=False, depthwise=True)
        self.bn2 = nn.BatchNorm2d(mid_chs, **norm_kwargs)
        self.act2 = act_layer(inplace=True)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio, reduced_base_chs=in_channel, 
                act_layer=act_layer)
        else:
            self.se = nn.Identity()

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(mid_chs, out_channel, 1, padding=pad_type, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel, **norm_kwargs)
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


def identity(x, inplace:bool=True) :
    return x

def dwconv3x3_block(in_channels, out_channels, stride=1, padding='same', dilation=1, bias=False, bn_eps=1e-5, activation=nn.ReLU):
    return DepthwiseSeparableConv(
        in_chs=in_channels, out_chs=out_channels, 
        dw_kernel_size=3, stride=stride, noskip=True,
        pad_type=padding, act_layer=activation, 
        norm_kwargs={'eps': bn_eps}, se_ratio=None
    )


def dwconv5x5_block(in_channels, out_channels, stride=1, padding='same', dilation=1, bias=False, bn_eps=1e-5, activation=nn.ReLU):
    return DepthwiseSeparableConv(
        in_chs=in_channels, out_chs=out_channels, 
        dw_kernel_size=5, stride=stride, noskip=True,
        pad_type=padding, act_layer=activation, 
        norm_kwargs={'eps': bn_eps}, se_ratio=None
    )
