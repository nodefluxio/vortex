'''
includes mixed convolutions
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import List, Dict, Tuple
from .constants import BN_ARGS_PT, BN_ARGS_TF
from .activations import sigmoid


# Set to True if exporting a model with Same padding via ONNX
_EXPORTABLE = False


def _is_static_pad(kernel_size, stride=1, dilation=1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


def _get_padding(kernel_size, stride=1, dilation=1, **_):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

@torch.jit.script
def _calc_same_pad(i : int, k : int, s : int, d : int):
    return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

@torch.jit.script
def _same_pad_arg(input_size : List[int], kernel_size : List[int], stride : Tuple[int,int], dilation : Tuple[int,int]) -> torch.Tensor:
    ih = input_size[0] 
    iw = input_size[1]
    kh = kernel_size[0]
    kw = kernel_size[1]
    pad_h = _calc_same_pad(ih, kh, stride[0], dilation[0])
    pad_w = _calc_same_pad(iw, kw, stride[1], dilation[1])
    # pad_arg = [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
    pad_arg = torch.tensor([pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
    return pad_arg


def _split_channels(num_chan, num_groups):
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split


class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        pad_arg = _same_pad_arg(x.size()[-2:], self.weight.size()[-2:], self.stride, self.dilation)
        pad_arg = [int(pad_arg[0]), int(pad_arg[1]), int(pad_arg[2]), int(pad_arg[3])]
        x = F.pad(x, pad_arg)
        return F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dSameExport(nn.Conv2d):
    """ ONNX export friendly Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSameExport, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.pad = None
        self.pad_input_size = (0, 0)

    def forward(self, x):
        input_size = x.size()[-2:]
        if self.pad is None:
            pad_arg = _same_pad_arg(input_size, self.weight.size()[-2:], self.stride, self.dilation)
            self.pad = nn.ZeroPad2d(pad_arg)
            self.pad_input_size = input_size
        else:
            assert self.pad_input_size == input_size

        x = self.pad(x)
        return F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == 'same':
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if _is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = _get_padding(kernel_size, **kwargs)
                return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)
            else:
                # dynamic padding
                if _EXPORTABLE:
                    return Conv2dSameExport(in_chs, out_chs, kernel_size, **kwargs)
                else:
                    return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
        elif padding == 'valid':
            # 'VALID' padding, same as padding=0
            return nn.Conv2d(in_chs, out_chs, kernel_size, padding=0, **kwargs)
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = _get_padding(kernel_size, **kwargs)
            return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)
    else:
        # padding was specified as a number or pair
        return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)

## workaround for torch.jit.script
class MixedConv2dModule2(nn.Module) :
    def __init__(self, kernel_size, in_splits, out_splits, stride=1, dilated=False, depthwise=False, padding='', **kwargs):
        super().__init__()
        assert(len(kernel_size)==2)
        conv_modules = []
        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            d = 1
            # FIXME make compat with non-square kernel/dilations/strides
            if stride == 1 and dilated:
                d, k = (k - 1) // 2, 3
            conv_groups = out_ch if depthwise else 1
            conv_modules.append(
                conv2d_pad(
                    in_ch, out_ch, k, stride=stride,
                    padding=padding, dilation=d, groups=conv_groups, **kwargs)
            )
        self.c0, self.c1 = conv_modules
    
    def forward(self, x : List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x0, x1 = x[0], x[1]
        return self.c0(x0), self.c1(x1)

class MixedConv2dModule3(nn.Module) :
    def __init__(self, kernel_size, in_splits, out_splits, stride=1, dilated=False, depthwise=False, padding='', **kwargs):
        super().__init__()
        assert(len(kernel_size)==3)
        conv_modules = []
        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            d = 1
            # FIXME make compat with non-square kernel/dilations/strides
            if stride == 1 and dilated:
                d, k = (k - 1) // 2, 3
            conv_groups = out_ch if depthwise else 1
            conv_modules.append(
                conv2d_pad(
                    in_ch, out_ch, k, stride=stride,
                    padding=padding, dilation=d, groups=conv_groups, **kwargs)
            )
        self.c0, self.c1, self.c2 = conv_modules
    
    def forward(self, x : List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x0, x1, x2 = x[0], x[1], x[2]
        return self.c0(x0), self.c1(x1), self.c2(x2)

class MixedConv2dModule4(nn.Module) :
    def __init__(self, kernel_size, in_splits, out_splits, stride=1, dilated=False, depthwise=False, padding='', **kwargs):
        super().__init__()
        assert(len(kernel_size)==4)
        conv_modules = []
        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            d = 1
            # FIXME make compat with non-square kernel/dilations/strides
            if stride == 1 and dilated:
                d, k = (k - 1) // 2, 3
            conv_groups = out_ch if depthwise else 1
            conv_modules.append(
                conv2d_pad(
                    in_ch, out_ch, k, stride=stride,
                    padding=padding, dilation=d, groups=conv_groups, **kwargs)
            )
        self.c0, self.c1, self.c2, self.c3 = conv_modules
    
    def forward(self, x : List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x0, x1, x2, x3 = x[0], x[1], x[2], x[3]
        return self.c0(x0), self.c1(x1), self.c2(x2), self.c3(x3)

class MixedConv2dModule5(nn.Module) :
    def __init__(self, kernel_size, in_splits, out_splits, stride=1, dilated=False, depthwise=False, padding='', **kwargs):
        super().__init__()
        assert(len(kernel_size)==5)
        conv_modules = []
        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            d = 1
            # FIXME make compat with non-square kernel/dilations/strides
            if stride == 1 and dilated:
                d, k = (k - 1) // 2, 3
            conv_groups = out_ch if depthwise else 1
            conv_modules.append(
                conv2d_pad(
                    in_ch, out_ch, k, stride=stride,
                    padding=padding, dilation=d, groups=conv_groups, **kwargs)
            )
        self.c0, self.c1, self.c2, self.c3, self.c4 = conv_modules
    
    def forward(self, x : List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x0, x1, x2, x3, x4 = x[0], x[1], x[2], x[3], x[4]
        return self.c0(x0), self.c1(x1), self.c2(x2), self.c3(x3), self.c4(x4)

class MixedConv2d(nn.Module):
    """ Mixed Grouped Convolution
    Based on MDConv and GroupedConv in MixNet impl:
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding='', dilated=False, depthwise=False, **kwargs):
        super(MixedConv2d, self).__init__()

        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        # self.conv_modules = []
        # for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
        #     d = 1
        #     # FIXME make compat with non-square kernel/dilations/strides
        #     if stride == 1 and dilated:
        #         d, k = (k - 1) // 2, 3
        #     conv_groups = out_ch if depthwise else 1
        #     # use add_module to keep key space clean
        #     # self.add_module(
        #     #     str(idx),
        #     #     conv2d_pad(
        #     #         in_ch, out_ch, k, stride=stride,
        #     #         padding=padding, dilation=d, groups=conv_groups, **kwargs)
        #     # )
        #     self.conv_modules.append(
        #         conv2d_pad(
        #             in_ch, out_ch, k, stride=stride,
        #             padding=padding, dilation=d, groups=conv_groups, **kwargs)
        #     )
        # self.n_modules = len(self.conv_modules)
        # self.conv_modules = nn.ModuleList(self.conv_modules)
        if len(kernel_size)==2 :
            self.conv_modules = MixedConv2dModule2(
                kernel_size=kernel_size,in_splits=in_splits,
                out_splits=out_splits,stride=stride,
                dilated=dilated,depthwise=depthwise,
                padding=padding,**kwargs
            )
        elif len(kernel_size)==3 :
            self.conv_modules = MixedConv2dModule3(
                kernel_size=kernel_size,in_splits=in_splits,
                out_splits=out_splits,stride=stride,
                dilated=dilated,depthwise=depthwise,
                padding=padding,**kwargs
            )
        elif len(kernel_size)==4 :
            self.conv_modules = MixedConv2dModule4(
                kernel_size=kernel_size,in_splits=in_splits,
                out_splits=out_splits,stride=stride,
                dilated=dilated,depthwise=depthwise,
                padding=padding,**kwargs
            )
        elif len(kernel_size)==5 :
            self.conv_modules = MixedConv2dModule5(
                kernel_size=kernel_size,in_splits=in_splits,
                out_splits=out_splits,stride=stride,
                dilated=dilated,depthwise=depthwise,
                padding=padding,**kwargs
            )
        else :
            raise NotImplementedError('mixed conv {} module has not been implemented yet'.format(len(kernel_size)))
        self.splits = in_splits
        print(self.conv_modules)

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        # x_out = [c(x) for x, c in zip(x_split, self._modules.values())]
        # x_out = [c(x) for x, c in zip(x_split, self.conv_modules)]
        # x_out = []
        # for i in range(len(self.n_modules)) :
        #     x_ = self.conv_modules[i](x_split[i])
        #     x_out.append(x_)
        x_out = self.conv_modules(x_split)
        x = torch.cat(x_out, 1)
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


def identity(x, inplace:bool=True) :
    return x

class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, pad_type='', act_fn=F.relu, bn_args=BN_ARGS_PT):
        super(ConvBnAct, self).__init__()
        assert stride in [1, 2]
        self.act_fn = act_fn if act_fn is not None else identity
        self.conv = select_conv2d(in_chs, out_chs, kernel_size, stride=stride, padding=pad_type)
        self.bn1 = nn.BatchNorm2d(out_chs, **bn_args)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act_fn(x, inplace=True)
        return x


class DepthwiseSeparableConv(nn.Module):
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks with an expansion
    factor of 1.0. This is an alternative to having a IR with optional first pw conv.
    """
    def __init__(self, in_chs, out_chs, dw_kernel_size=3,
                 stride=1, pad_type='', act_fn=F.relu, noskip=False,
                 pw_kernel_size=1, pw_act=False,
                 se_ratio=0., se_gate_fn=sigmoid,
                 bn_args=BN_ARGS_PT, drop_connect_rate=0.):
        super(DepthwiseSeparableConv, self).__init__()
        assert stride in [1, 2]
        self.has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act  # activation after point-wise conv
        self.act_fn = act_fn
        self.drop_connect_rate = drop_connect_rate

        self.conv_dw = select_conv2d(
            in_chs, in_chs, dw_kernel_size, stride=stride, padding=pad_type, depthwise=True)
        self.bn1 = nn.BatchNorm2d(in_chs, **bn_args)

        # Squeeze-and-excitation
        if self.has_se:
            self.se = SqueezeExcite(
                in_chs, reduce_chs=max(1, int(in_chs * se_ratio)), act_fn=act_fn, gate_fn=se_gate_fn)
        else :
            self.se = torch.nn.Identity()

        self.conv_pw = select_conv2d(in_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = nn.BatchNorm2d(out_chs, **bn_args)

    def forward(self, x : torch.Tensor):
        residual = x

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act_fn(x, inplace=True)

        # if self.has_se:
            # x = self.se(x)
        x = self.se(x)

        x = self.conv_pw(x)
        x = self.bn2(x)
        if self.has_pw_act:
            x = self.act_fn(x, inplace=True)

        if self.has_residual:
            if self.drop_connect_rate > 0.:
                x = drop_connect(x, self.training, self.drop_connect_rate)
            x += residual
        return x


# helper method
def select_conv2d(in_chs, out_chs, kernel_size, **kwargs):
    assert 'groups' not in kwargs  # only use 'depthwise' bool arg
    if isinstance(kernel_size, list):
        # We're going to use only lists for defining the MixedConv2d kernel groups,
        # ints, tuples, other iterables will continue to pass to normal conv and specify h, w.
        return MixedConv2d(in_chs, out_chs, kernel_size, **kwargs)
    else:
        depthwise = kwargs.pop('depthwise', False)
        groups = out_chs if depthwise else 1
        return conv2d_pad(in_chs, out_chs, kernel_size, groups=groups, **kwargs)


def conv1x1_block(in_channels, out_channels, stride=1, padding=0, groups=1, bias=False, use_bn=True, bn_eps=1e-5, activation=F.relu) :
    return ConvBnAct(
        in_chs=in_channels, out_chs=out_channels, 
        kernel_size=1, stride=stride, act_fn=activation,
        bn_args={'eps': bn_eps}, pad_type=padding
    )


def conv3x3_block(in_channels, out_channels, stride=1, padding=1, dilation=1, group=1, bias=False, use_bn=True, bn_eps=1e-5, activation=F.relu) :
    return ConvBnAct(
        in_chs=in_channels, out_chs=out_channels,
        kernel_size=3, stride=stride, act_fn=activation,
        bn_args={'eps': bn_eps}, pad_type=padding
    )


def dwconv3x3_block(in_channels, out_channels, stride=1, padding='same', dilation=1, bias=False, bn_eps=1e-5, activation=F.relu) :
    return DepthwiseSeparableConv(
        in_chs=in_channels, out_chs=out_channels, 
        dw_kernel_size=3, stride=stride, noskip=True,
        pad_type=padding, act_fn=activation, 
        bn_args={'eps': bn_eps}, se_ratio=None
    )


def dwconv5x5_block(in_channels, out_channels, stride=1, padding='same', dilation=1, bias=False, bn_eps=1e-5, activation=F.relu) :
    return DepthwiseSeparableConv(
        in_chs=in_channels, out_chs=out_channels, 
        dw_kernel_size=5, stride=stride, noskip=True,
        pad_type=padding, act_fn=activation, 
        bn_args={'eps': bn_eps}, se_ratio=None
    )


def conv1x1(in_channels,
            out_channels,
            stride=1,
            groups=1,
            bias=False):
    """
    Convolution 1x1 layer.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        groups=groups,
        bias=bias)

