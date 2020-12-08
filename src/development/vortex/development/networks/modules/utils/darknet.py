import enforce
import torch
import torch.nn as nn
import numpy as np

from typing import Tuple, List


__all__ = [
    'DarknetResidual',
    'DarknetTinyBlock',
    'activation_fn',
    'darknet_maxpool',
    'darknet_conv',
    'yolo_feature_maps',
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
def yolo_feature_maps(input_size: int, backbone_stages: Tuple[int, int, int] = (3,4,5)) -> List[Tuple[int, int]]:
    assert len(backbone_stages) == 3, "'backbone_stages' ({}) should have length of 3, got".format(backbone_stages)
    s3, s4, s5 = (2**x for x in backbone_stages)
    return [
        (input_size//s3, input_size//s3),
        (input_size//s4, input_size//s4),
        (input_size//s5, input_size//s5),
    ]



### ============ darknet weight converter ============ ###

def _convert_module_weight(weights, module, ptr):
    param_names = []
    if isinstance(module, nn.BatchNorm2d):
        param_names.extend(['bias', 'weight', 'running_mean', 'running_var'])
    elif isinstance(module, (nn.Conv2d, nn.Linear)):
        if getattr(module, 'bias') is not None:
            param_names.append('bias')
        param_names.append('weight')
    else:
        raise RuntimeError("unknown module")

    for pname in param_names:
        module_param = getattr(module, pname)
        n = module_param.numel()
        param = torch.from_numpy(weights[ptr: ptr+n]).view_as(module_param)
        module_param.data.copy_(param)
        ptr += n
    return ptr


def load_darknet_weight(model, weights):
    with open(weights, "rb") as fp:
        version = np.fromfile(fp, dtype=np.int32, count=3)      # weight version
        if (version[0]*10 + version[1]) >= 2 and version[0] < 1000 and version[1] < 1000:
            dtype = np.int64
        else:
            dtype = np.int32
        seen = np.fromfile(fp, dtype=dtype, count=1)[0]
        weights = np.fromfile(fp, dtype=np.float32)

    total_params = len(weights)
    offset = 0

    before_upsample = None
    module_iter = model.modules()
    while offset < total_params:
        module = next(module_iter)
        if isinstance(module, nn.Conv2d):
            bn_module = next(module_iter)
            if isinstance(bn_module, nn.BatchNorm2d):
                offset = _convert_module_weight(weights, bn_module, offset)
            offset = _convert_module_weight(weights, module, offset)

            if (not isinstance(bn_module, nn.BatchNorm2d) and before_upsample != None
                    and offset < total_params):
                offset = _convert_module_weight(weights, before_upsample[1], offset)
                offset = _convert_module_weight(weights, before_upsample[0], offset)
                before_upsample = None
        elif isinstance(module, nn.Linear):
            offset = _convert_module_weight(weights, module, offset)
        elif isinstance(module, nn.Sequential):
            if isinstance(module[0], nn.Sequential) and isinstance(module[1], nn.Upsample):
                before_upsample = module[0]
                ## this is a workaround for different placement of modules after upsample
                ## TODO: fix model YOLOV3 definition to have the same order as in darknet
                while not isinstance(next(module_iter), nn.Upsample):
                    pass
    assert offset == len(weights), "total number of parameters in model definition " \
        "and in the weight is not the same."
    return model.state_dict()
