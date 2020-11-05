""" Padding Helpers

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
import torch.nn.functional as F

from typing import List, Tuple


# Calculate symmetric padding for a convolution
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


# Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(float(x) / s) - 1) * s + (k - 1) * d + 1 - x, 0)


# Can SAME padding for given args be done statically?
def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


# Dynamically pad input x with 'SAME' padding for conv with specified args
def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0):
    ih, iw = x.size()[-2:]
    pad_h = get_same_padding(int(ih), int(k[0]), int(s[0]), int(d[0]))
    pad_w = get_same_padding(int(iw), int(k[1]), int(s[1]), int(d[1]))
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x


def get_padding_value(padding, kernel_size, **kwargs) -> Tuple[Tuple, bool]:
    dynamic = False
    ## for any string padding, the padding will be calculated for you, one of three ways
    if isinstance(padding, str): 
        padding = padding.lower()
        ## TF compatible 'SAME' padding
        # has a performance and GPU memory allocation impact
        if padding == 'same': 
            if is_static_pad(kernel_size, **kwargs): # static case, no extra overhead
                padding = get_padding(kernel_size, **kwargs)
            else: # dynamic 'SAME' padding, has runtime/GPU memory overhead
                padding = 0
                dynamic = True
        elif padding == 'valid': # 'VALID' padding, same as padding=0
            padding = 0
        else: # Default to PyTorch style 'same'-ish symmetric padding
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic
