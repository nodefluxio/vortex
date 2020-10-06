""" Layers compilation

Hacked together by / Copyright 2020 Ross Wightman
Edited by Vortex Team
"""

import types
import functools
import warnings
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv2d import create_conv2d, tup_pair
from .activations import sigmoid, get_act_layer, create_act_layer
from .arch_utils import make_divisible
from .padding import pad_same

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


def get_norm_act_layer(layer_class):
    layer_class = layer_class.replace('_', '').lower()
    if layer_class.startswith("batchnorm"):
        layer = BatchNormAct2d
    elif layer_class.startswith("groupnorm"):
        layer = GroupNormAct
    elif layer_class == "evonormbatch":
        layer = EvoNormBatch2d
    elif layer_class == "evonormsample":
        layer = EvoNormSample2d
    else:
        assert False, "Invalid norm_act layer (%s)" % layer_class
    return layer

class BatchNormAct2d(nn.BatchNorm2d):
    """BatchNorm + Activation

    This module performs BatchNorm + Activation in a manner that will remain backwards
    compatible with weights trained with separate bn, act. This is why we inherit from BN
    instead of composing it as a .bn member.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 apply_act=True, act_layer=nn.ReLU, inplace=True, drop_block=None):
        super(BatchNormAct2d, self).__init__(
            num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        if isinstance(act_layer, str):
            act_layer = get_act_layer(act_layer)
        if act_layer is not None and apply_act:
            act_args = dict(inplace=True) if inplace else {}
            self.act = act_layer(**act_args)
        else:
            self.act = None

    def _forward_jit(self, x):
        """ A cut & paste of the contents of the PyTorch BatchNorm2d forward function
        """
        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        x = F.batch_norm(
                x, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)
        return x

    @torch.jit.ignore
    def _forward_python(self, x):
        return super(BatchNormAct2d, self).forward(x)

    def forward(self, x):
        # FIXME cannot call parent forward() and maintain jit.script compatibility?
        if torch.jit.is_scripting():
            x = self._forward_jit(x)
        else:
            x = self._forward_python(x)
        if self.act is not None:
            x = self.act(x)
        return x


class GroupNormAct(nn.GroupNorm):

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True,
                 apply_act=True, act_layer=nn.ReLU, inplace=True, drop_block=None):
        super(GroupNormAct, self).__init__(num_groups, num_channels, eps=eps, affine=affine)
        if isinstance(act_layer, str):
            act_layer = get_act_layer(act_layer)
        if act_layer is not None and apply_act:
            self.act = act_layer(inplace=inplace)
        else:
            self.act = None

    def forward(self, x):
        x = F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        if self.act is not None:
            x = self.act(x)
        return x


class EvoNormBatch2d(nn.Module):
    def __init__(self, num_features, apply_act=True, momentum=0.1, eps=1e-5, drop_block=None):
        super(EvoNormBatch2d, self).__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.momentum = momentum
        self.eps = eps
        param_shape = (1, num_features, 1, 1)
        self.weight = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(param_shape), requires_grad=True)
        if apply_act:
            self.v = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.apply_act:
            nn.init.ones_(self.v)

    def forward(self, x):
        assert x.dim() == 4, 'expected 4D input'
        x_type = x.dtype
        if self.training:
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            n = x.numel() / x.shape[1]
            self.running_var.copy_(
                var.detach() * self.momentum * (n / (n - 1)) + self.running_var * (1 - self.momentum))
        else:
            var = self.running_var

        if self.apply_act:
            v = self.v.to(dtype=x_type)
            d = x * v + (x.var(dim=(2, 3), unbiased=False, keepdim=True) + self.eps).sqrt().to(dtype=x_type)
            d = d.max((var + self.eps).sqrt().to(dtype=x_type))
            x = x / d
        return x * self.weight + self.bias


class EvoNormSample2d(nn.Module):
    def __init__(self, num_features, apply_act=True, groups=8, eps=1e-5, drop_block=None):
        super(EvoNormSample2d, self).__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.groups = groups
        self.eps = eps
        param_shape = (1, num_features, 1, 1)
        self.weight = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(param_shape), requires_grad=True)
        if apply_act:
            self.v = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.apply_act:
            nn.init.ones_(self.v)

    def forward(self, x):
        assert x.dim() == 4, 'expected 4D input'
        B, C, H, W = x.shape
        assert C % self.groups == 0
        if self.apply_act:
            n = x * (x * self.v).sigmoid()
            x = x.reshape(B, self.groups, -1)
            x = n.reshape(B, self.groups, -1) / (x.var(dim=-1, unbiased=False, keepdim=True) + self.eps).sqrt()
            x = x.reshape(B, C, H, W)
        return x * self.weight + self.bias


_NORM_ACT_TYPES = {BatchNormAct2d, GroupNormAct, EvoNormBatch2d, EvoNormSample2d}
_NORM_ACT_REQUIRES_ARG = {BatchNormAct2d, GroupNormAct}

def convert_norm_act_type(norm_layer, act_layer, norm_kwargs=None):
    assert isinstance(norm_layer, (type, str,  types.FunctionType, functools.partial))
    assert act_layer is None or isinstance(act_layer, (type, str, types.FunctionType, functools.partial))

    norm_act_args = norm_kwargs.copy() if norm_kwargs else {}
    if isinstance(norm_layer, str):
        norm_act_layer = get_norm_act_layer(norm_layer)
    elif norm_layer in _NORM_ACT_TYPES:
        norm_act_layer = norm_layer
    elif isinstance(norm_layer,  (types.FunctionType, functools.partial)):
        # assuming this is a lambda/fn/bound partial that creates norm_act layer
        norm_act_layer = norm_layer
    else:
        type_name = norm_layer.__name__.lower()
        if type_name.startswith('batchnorm'):
            norm_act_layer = BatchNormAct2d
        elif type_name.startswith('groupnorm'):
            norm_act_layer = GroupNormAct
        else:
            warnings.warn("No equivalent norm_act layer for {}".format(type_name))
            norm_act_layer = norm_layer
    if norm_act_layer in _NORM_ACT_REQUIRES_ARG:
        # Must pass `act_layer` through for backwards compat where `act_layer=None` implies no activation.
        # In the future, may force use of `apply_act` with `act_layer` arg bound to relevant NormAct types
        # It is intended that functions/partial does not trigger this, they should define act.
        norm_act_args.update(dict(act_layer=act_layer))
    return norm_act_layer, norm_act_args


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
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding='', dilation=1, groups=1,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, act_layer=nn.ReLU, apply_act=True,
                 drop_block=None, aa_layer=None):
        super(ConvBnAct, self).__init__()
        use_aa = aa_layer is not None

        self.conv = create_conv2d(in_channels, out_channels, kernel_size, padding=padding, 
            stride=1 if use_aa else stride, dilation=dilation, groups=groups, bias=False)

        # NOTE for backwards compatibility with models that use separate norm and act layer definitions
        norm_act_layer, norm_act_args = convert_norm_act_type(norm_layer, act_layer, norm_kwargs)
        self.bn = norm_act_layer(out_channels, apply_act=apply_act, drop_block=drop_block, **norm_act_args)
        self.aa = aa_layer(channels=out_channels) if stride == 2 and use_aa else None

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.aa is not None:
            x = self.aa(x)
        return x


class SEModule(nn.Module):

    def __init__(self, channels, reduction=16, act_layer=nn.ReLU, min_channels=8, reduction_channels=None,
                 gate_layer='sigmoid'):
        super(SEModule, self).__init__()
        reduction_channels = reduction_channels or max(channels // reduction, min_channels)
        self.fc1 = nn.Conv2d(channels, reduction_channels, kernel_size=1, bias=True)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(reduction_channels, channels, kernel_size=1, bias=True)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)


class EffectiveSEModule(nn.Module):
    """ 'Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """
    def __init__(self, channels, gate_layer='hard_sigmoid'):
        super(EffectiveSEModule, self).__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.gate = create_act_layer(gate_layer, inplace=True)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.gate(x_se)


def avg_pool2d_same(x, kernel_size: List[int], stride: List[int], padding: List[int] = (0, 0),
                    ceil_mode: bool = False, count_include_pad: bool = True):
    # FIXME how to deal with count_include_pad vs not for external padding?
    x = pad_same(x, kernel_size, stride)
    return F.avg_pool2d(x, kernel_size, stride, (0, 0), ceil_mode, count_include_pad)


class AvgPool2dSame(nn.AvgPool2d):
    """ Tensorflow like 'SAME' wrapper for 2D average pooling
    """
    def __init__(self, kernel_size: int, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        kernel_size = tup_pair(kernel_size)
        stride = tup_pair(stride)
        super(AvgPool2dSame, self).__init__(kernel_size, stride, (0, 0), ceil_mode, count_include_pad)

    def forward(self, x):
        return avg_pool2d_same(
            x, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)


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


def drop_block_2d(x, drop_prob: float = 0.1, block_size: int = 7,  gamma_scale: float = 1.0,
                  with_noise: bool = False, inplace: bool = False, batchwise: bool = False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. This layer has been tested on a few training
    runs with success, but needs further validation and possibly optimization for lower runtime impact.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    # seed_drop_rate, the gamma parameter
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / (
        (W - block_size + 1) * (H - block_size + 1))

    # Forces the block to be inside the feature map.
    w_i, h_i = torch.meshgrid(torch.arange(W).to(x.device), torch.arange(H).to(x.device))
    valid_block = ((w_i >= clipped_block_size // 2) & (w_i < W - (clipped_block_size - 1) // 2)) & \
                  ((h_i >= clipped_block_size // 2) & (h_i < H - (clipped_block_size - 1) // 2))
    valid_block = torch.reshape(valid_block, (1, 1, H, W)).to(dtype=x.dtype)

    if batchwise:
        # one mask for whole batch, quite a bit faster
        uniform_noise = torch.rand((1, C, H, W), dtype=x.dtype, device=x.device)
    else:
        uniform_noise = torch.rand_like(x)
    block_mask = ((2 - gamma - valid_block + uniform_noise) >= 1).to(dtype=x.dtype)
    block_mask = -F.max_pool2d(
        -block_mask,
        kernel_size=clipped_block_size,  # block_size,
        stride=1,
        padding=clipped_block_size // 2)

    if with_noise:
        normal_noise = torch.randn((1, C, H, W), dtype=x.dtype, device=x.device) if batchwise else torch.randn_like(x)
        if inplace:
            x.mul_(block_mask).add_(normal_noise * (1 - block_mask))
        else:
            x = x * block_mask + normal_noise * (1 - block_mask)
    else:
        normalize_scale = (block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-7)).to(x.dtype)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


def drop_block_fast_2d(x: torch.Tensor, drop_prob: float = 0.1, block_size: int = 7,
        gamma_scale: float = 1.0, with_noise: bool = False, inplace: bool = False, batchwise: bool = False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. Simplied from above without concern for valid
    block mask at edges.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / (
            (W - block_size + 1) * (H - block_size + 1))

    if batchwise:
        # one mask for whole batch, quite a bit faster
        block_mask = torch.rand((1, C, H, W), dtype=x.dtype, device=x.device) < gamma
    else:
        # mask per batch element
        block_mask = torch.rand_like(x) < gamma
    block_mask = F.max_pool2d(
        block_mask.to(x.dtype), kernel_size=clipped_block_size, stride=1, padding=clipped_block_size // 2)

    if with_noise:
        normal_noise = torch.randn((1, C, H, W), dtype=x.dtype, device=x.device) if batchwise else torch.randn_like(x)
        if inplace:
            x.mul_(1. - block_mask).add_(normal_noise * block_mask)
        else:
            x = x * (1. - block_mask) + normal_noise * block_mask
    else:
        block_mask = 1 - block_mask
        normalize_scale = (block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-7)).to(dtype=x.dtype)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


class DropBlock2d(nn.Module):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf
    """
    def __init__(self,
                 drop_prob=0.1,
                 block_size=7,
                 gamma_scale=1.0,
                 with_noise=False,
                 inplace=False,
                 batchwise=False,
                 fast=True):
        super(DropBlock2d, self).__init__()
        self.drop_prob = drop_prob
        self.gamma_scale = gamma_scale
        self.block_size = block_size
        self.with_noise = with_noise
        self.inplace = inplace
        self.batchwise = batchwise
        self.fast = fast  # FIXME finish comparisons of fast vs not

    def forward(self, x):
        if not self.training or not self.drop_prob:
            return x
        if self.fast:
            return drop_block_fast_2d(
                x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise, self.inplace, self.batchwise)
        else:
            return drop_block_2d(
                x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise, self.inplace, self.batchwise)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    random_tensor = keep_prob + torch.rand((x.size()[0], 1, 1, 1), dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class FastAdaptiveAvgPool2d(nn.Module):
    def __init__(self, flatten=False):
        super(FastAdaptiveAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        return x.mean((2, 3)) if self.flatten else x.mean((2, 3), keepdim=True)


class ClassifierHead(nn.Module):
    """Classifier head w/ configurable global pooling and dropout."""

    def __init__(self, in_chs, num_classes, pool_type='avg', drop_rate=0.):
        super(ClassifierHead, self).__init__()

        self.flatten = True
        if pool_type == '':
            self.global_pool = nn.Identity()
        elif pool_type == 'fast':
            self.global_pool = FastAdaptiveAvgPool2d(flatten=True)
            self.flatten = False
        elif pool_type == 'avg':
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)
        else:
            raise RuntimeError("Unknown 'pool_type' of {}".format(pool_type))

        self.dropout = nn.Dropout(drop_rate) if drop_rate else None

        ## classifier
        if num_classes <= 0:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(in_chs, num_classes)

    def forward(self, x):
        x = self.global_pool(x)
        if self.flatten:
            x = x.flatten(1)
        if self.dropout:
            x = self.dropout(x)
        x = self.fc(x)
        return x
