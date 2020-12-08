"""
https://github.com/pytorch/vision/blob/0156d58ec867590b1c78fe1bc834c7da9afdf46a/torchvision/ops/feature_pyramid_network.py#L10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Union

def basic_lateral_block(in_channel : int, out_channel : int, bias : bool=True) :
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 1, 1, bias=bias),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

class BasicLateralBlock(nn.Module) :
    def __init__(self, in_channel : int, out_channel : int, bias : bool=True) :
        super(BasicLateralBlock, self).__init__()
        self.conv = basic_lateral_block(in_channel, out_channel, bias)
    def forward(self, feature) :
        return self.conv(feature)

class BasicTopDownBlock(nn.Module) :
    def __init__(self, in_channel : int, out_channel : int) :
        super(BasicTopDownBlock, self).__init__()
    def forward(self, lateral_features : torch.Tensor, upper_pyramid_features : torch.Tensor) :
        return lateral_features + nn.functional.interpolate(upper_pyramid_features,scale_factor=2)

class LastLevelP6P7(nn.Module):
    ## from detectron2
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from
    C5 feature.
    """

    def __init__(self, input_channels : int, output_channels : Union[int,list]):
        super().__init__()
        self.num_levels = 2
        if isinstance(output_channels, int) :
            output_channels = [output_channels] * self.num_levels
        assert len(output_channels) == self.num_levels
        self.p6 = nn.Conv2d(input_channels, output_channels[0], 3, 2, 1)
        self.p7 = nn.Conv2d(output_channels[0], output_channels[1], 3, 2, 1)

    def forward(self, c5):
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        return p6, p7

class FPNProto(nn.Module) :
    """
    actually lateral + top-down only (no bottom-up)
    ## TODO : better name (?)
    """
    def __init__(self, input_channels : List[int], output_channels : Union[List[int]], extra_block_factory=None, lateral_block_factory=BasicLateralBlock, top_down_factory=BasicTopDownBlock) :
        super(FPNProto, self).__init__()
        n_inputs  = len(input_channels)
        assert n_inputs==3, "current implementation only support 3 stage input, got %s" %n_inputs
        if isinstance(output_channels, int) :
            output_channels = [output_channels] * len(input_channels)
        n_outputs = len(output_channels)
        assert n_outputs >= n_inputs
        self.input_channels = input_channels
        self.output_channels = output_channels
        assert hasattr(lateral_block_factory, '__init__')
        assert hasattr(lateral_block_factory.__init__, '__annotations__')
        assert all([field in lateral_block_factory.__init__.__annotations__.keys() for field in ['in_channel', 'out_channel']])
        for i in range(n_inputs) :
            in_channel, out_channel = input_channels[i], output_channels[i]
            self.add_module('lateral%s' %(i+3), lateral_block_factory(in_channel=in_channel, out_channel=out_channel))
            self.add_module('topdown%s' %(i+3), top_down_factory(in_channel=in_channel, out_channel=out_channel))
        last_in_channels = input_channels[-1]
        self.extra_conv = None
        if extra_block_factory is not None :
            assert hasattr(extra_block_factory, '__init__')
            assert hasattr(extra_block_factory.__init__, '__annotations__')
            annotations = extra_block_factory.__init__.__annotations__
            assert all([arg in ['input_channels', 'output_channels'] for arg in annotations.keys()])
            if n_outputs == n_inputs:
                output_channels = output_channels[-1]
            else:
                output_channels = output_channels[n_inputs:]
            ## TODO : check (or enforce) `output_channels` to accept Union[int,list]
            self.extra_conv = extra_block_factory(input_channels=last_in_channels,output_channels=output_channels)
        elif n_outputs > n_inputs :
            import warnings
            warnings.warn("ignoring output_channles[{}:]".format(n_inputs))

    def forward(self, c3 : torch.Tensor, c4 : torch.Tensor, c5 : torch.Tensor) :
        """
        extra_conv  ->          -> out3
           |^                    + (upsample)
        feature2    -> lateral2 -> out2
                                 + (upsample)
        featrue1    -> lateral1 -> out1
                                 + (upsample)
        feature0    -> lateral0 -> out0
        """
        lateral3_result = self.lateral3(c3)
        lateral4_result = self.lateral4(c4)
        lateral5_result = self.lateral5(c5) ## note (lateral 5 is actually top-layer if no extra blocs)
        ## TODO : make torch.script-able (?)
        top_features = None
        if not self.extra_conv is None :
            extra_features = self.extra_conv(c5)
            top_features = extra_features
            if isinstance(top_features, tuple) or isinstance(top_features, list):
                p6 = top_features[0]
            else :
                p6 = top_features
            p5 = self.topdown5(lateral5_result, p6)
        else :
            p5 = lateral5_result
        p4 = self.topdown4(lateral4_result, p5)
        p3 = self.topdown3(lateral3_result, p4)
        # up6 = nn.functional.interpolate(p6)
        # p5 = lateral5_result + up6
        # up5 = nn.functional.interpolate(p5)
        # p4 = lateral4_result + up5
        # up4 = nn.functional.interpolate(p4)
        # p3 = lateral3_result + up4
        return p3, p4, p5, top_features

from ....models.base_connector import BackbonePoolConnector

import torch
import torch.nn as nn

class FPNBackbone(BackbonePoolConnector) :
    """
    bottom-up + lateral + top-down
    """
    def __init__(self, pyramid_channels : Union[List[int],int], extra_block_factory=None, *args, **kwargs) :
        super(FPNBackbone, self).__init__(*args, **kwargs)
        backbone_channels = self.backbone.out_channels
        self.fpn = FPNProto(input_channels=backbone_channels[2:], output_channels=pyramid_channels, extra_block_factory=extra_block_factory)
        self.extra_block = extra_block_factory is not None
    
    def forward(self, x) :
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5, p67 = self.fpn(c3, c4, c5)
        ## TODO : return extra pyramid if extra blocks exists!
        if self.extra_block:
            return p3, p4, p5, p67
        else:
            return p3, p4, p5
