from types import FunctionType
import torch
import torch.nn as nn
from typing import Tuple, List
import enforce

__all__ = [
    'DarknetResidual',
    'DarknetTinyBlock',
    'activation_fn',
    'darknet_maxpool',
    'darknet_conv',
    'yolo_feature_maps',
    'intermediate_layer_codegen',
    'create_intermediate_layer'
]

act_map = {
    'leaky': 'nn.LeakyReLU',
    'prelu': 'nn.PReLU',
    'linear': 'nn.Identity'
    # TODO : add hardswish (?)
}


class DarknetResidual(nn.Module):
    """
    [convolutional]
    batch_normalize=1
    filters=filters[0]
    size=1
    stride=1
    pad=1
    activation=leaky

    [convolutional]
    batch_normalize=1
    filters=filters[1]
    size=3
    stride=1
    pad=1
    activation=leaky

    [shortcut]
    from=-3
    activation=linear
    """

    def __init__(self, in_channels: int, filters: Tuple[int, int],  *args, **kwargs):
        super(DarknetResidual, self).__init__()
        self.conv = nn.Sequential(
            darknet_conv(
                in_channels, filters[0], bn=True,
                kernel_size=1, pad=True, stride=1,
                activation='leaky'
            ),
            darknet_conv(
                filters[0], filters[1], bn=True,
                kernel_size=3, pad=True, stride=1,
                activation='leaky'
            )
        )

    def forward(self, x: torch.Tensor):
        return x + self.conv(x)


class DarknetTinyBlock(nn.Module):
    """
    [convolutional]
    batch_normalize=1
    filters=int
    size=3
    stride=1
    pad=1
    activation=leaky

    [maxpool]
    size=int
    stride=int
    """

    def __init__(self, in_channels: int, filters: int, pool_size=2, pool_stride=2, *args, **kwargs):
        super(DarknetTinyBlock, self).__init__()
        self.conv = nn.Sequential(
            darknet_conv(in_channels, filters, bn=True,
                         kernel_size=3, pad=True, stride=1,
                         activation='leaky'
                         ),
            darknet_maxpool(
                kernel_size=pool_size,
                stride=pool_stride
            )
        )

    def forward(self, x: torch.Tensor):
        return self.conv(x)


def activation_fn(act: str, *args, **kwargs):
    if act in act_map.keys():
        act = act_map[act]
    return eval('%s(*args,*kwargs)' % act)


@enforce.runtime_validation
def darknet_maxpool(kernel_size: int, stride: int):
    maxpool = nn.MaxPool2d(kernel_size=kernel_size,
                           stride=stride, padding=int((kernel_size - 1) // 2))
    if kernel_size == 2 and stride == 1:
        return nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            maxpool
        )
    else:
        return maxpool


@enforce.runtime_validation
def darknet_conv(in_channels: int, filters: int, bn: bool, kernel_size: int, pad: bool, activation: str, stride: int):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=((kernel_size-1)//2 if pad else 0),
            bias=not bn
        ),
        nn.BatchNorm2d(
            filters,
            momentum=0.1
        ) if bn else nn.Identity(),
        activation_fn(activation, 0.1, inplace=True)
    )


@enforce.runtime_validation
def yolo_feature_maps(input_size: int) -> List[Tuple[int, int]]:
    s3, s4, s5 = 2**3, 2**4, 2**5
    return [
        (input_size//s3, input_size//s3),
        (input_size//s4, input_size//s4),
        (input_size//s5, input_size//s5),
    ]
