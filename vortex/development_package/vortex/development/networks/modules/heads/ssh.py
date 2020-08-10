import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Union, Tuple, List

def conv_bn(in_channels, out_channels, kernel_size : int, stride=1, bias=False, relu=True) -> nn.Sequential :
    """
    create standard convolution with batchnorm + relu
    """
    return nn.Sequential(
        nn.Conv2d(int(in_channels), int(out_channels), kernel_size, stride, bias=bias, padding=(1,1)),
        nn.BatchNorm2d(int(out_channels)),
        nn.ReLU(inplace=True) if relu else nn.Identity()
    )

class SSHContextModule(nn.Module) :
    """
    torch module for SSH context module
    Reference : 
    Najibi, Mahyar Samangouei, PouyaChellappa, RamaDavis, Larry S. 
    SSH: Single Stage Headless Face Detector.
    arXiv:1708.03979v3. 2017
    """
    def __init__(self, in_channels : int, ctx_channels : int) :
        '''
        create SSH context module
        :in_channels: number of input channels
        :ctx_channels: number for detection channels
        '''
        super(SSHContextModule, self).__init__()
        ## note :  conv_bn includes relu
        self.conv1  = conv_bn(in_channels, ctx_channels, kernel_size=3, bias=True)
        self.conv2a = conv_bn(ctx_channels, ctx_channels, kernel_size=3, bias=True)
        self.conv2b = conv_bn(ctx_channels, ctx_channels, kernel_size=3, bias=True,relu=False)
        self.conv2c = conv_bn(ctx_channels, ctx_channels, kernel_size=3, bias=True)
    
    def forward(self, x) :
        '''forward operation :
                      | -> conv2a -----------> |
        x -> conv1 -> |                        | -> concat
                      | -> conv2b -> conv2c -> |
        '''
        x  = self.conv1(x)
        xa = self.conv2a(x)
        xb = self.conv2b(x)
        xc = self.conv2c(xb)
        return torch.cat((xa, xc), 1)

class SSHDetectionModule(nn.Module) :
    '''
    torch module for SSH detection module
    Reference : 
    Najibi, Mahyar Samangouei, PouyaChellappa, RamaDavis, Larry S. 
    SSH: Single Stage Headless Face Detector.
    arXiv:1708.03979v3. 2017
    '''
    def __init__(self, in_channels, out_channels=[4, 8]) :
        """
        create ssh detection module
        :in_channels: number of input channel
        :out_channels: number of channel for score & bbox
        """
        super(SSHDetectionModule, self).__init__()
        assert(len(out_channels)==2)
        self.conv1       = conv_bn(in_channels, in_channels/2, kernel_size=3, bias=True)
        self.context_mod = SSHContextModule(in_channels, in_channels/4)
        self.conv2a      = nn.Conv2d(in_channels, out_channels[0], kernel_size=1, bias=True)
        self.conv2b      = nn.Conv2d(in_channels, out_channels[1], kernel_size=1, bias=True)

    def forward(self, x) :
        """
        forward operation :
             | -> conv_relu -> |                      | -> conv1x1 -> out
        x -> | -> context   -> | -> concat -> ReLU -> | -> conv1x1 -> out
        """
        conv1_out = self.conv1(x)
        ctx_out   = self.context_mod(x)
        ctx_relu  = F.relu(torch.cat((conv1_out, ctx_out),1))
        return (
            self.conv2a(ctx_relu),
            self.conv2b(ctx_relu)
        )

class SSHLandmarkDetectionModule(nn.Module) :
    """
    torch module for SSH detection module with landmarks
    """
    def __init__(self, in_channels, out_channels=[4, 8, 20]) :
        """
        create ssh detection module
        :in_channels: number of input channel
        :out_channels: number of output channel
        :stride: number of stride for score & bbox & landmarks
        """
        super(SSHLandmarkDetectionModule, self).__init__()
        assert(len(out_channels)==3)
        self.conv1       = conv_bn(in_channels, in_channels/2, kernel_size=3, bias=True)
        self.context_mod = SSHContextModule(in_channels, in_channels/4)
        self.conv2a      = nn.Conv2d(in_channels, out_channels[0], kernel_size=1, bias=True)
        self.conv2b      = nn.Conv2d(in_channels, out_channels[1], kernel_size=1, bias=True)
        self.conv2c      = nn.Conv2d(in_channels, out_channels[2], kernel_size=1, bias=True)

    def forward(self, x) :
        """
        forward operation :
             | -> conv_relu -> |                      | -> conv1x1 -> out
        x -> | -> context   -> | -> concat -> ReLU -> | -> conv1x1 -> out
                                                      | -> conv1x1 -> out
        """
        conv1_out = self.conv1(x)
        ctx_out   = self.context_mod(x)
        ctx_relu  = F.relu(torch.cat((conv1_out, ctx_out),1))
        return (
            self.conv2a(ctx_relu),
            self.conv2b(ctx_relu),
            self.conv2c(ctx_relu)
        )

class SSHDetectionHead(nn.Module) :
    n_pyramid = 3
    def __init__(self, in_channels : Union[int,List[int]], n_anchors : int=2) :
        super(SSHDetectionHead, self).__init__()
        if isinstance(in_channels, int) :
            in_channels = [in_channels] * self.n_pyramid
        assert len(in_channels) == self.n_pyramid
        # self.det3 = SSHDetectionModule(in_channels[0])
        # self.det4 = SSHDetectionModule(in_channels[1])
        # self.det5 = SSHDetectionModule(in_channels[2])
        
        out_channels = [2,4]
        out_channels = [ch * n_anchors for ch in out_channels]

        assert all([in_channels[0]==ch for ch in in_channels])
        self.det_module = SSHDetectionModule(in_channels[0],out_channels=out_channels)

    def permute(self, cls_tensor, box_tensor) :
        n = cls_tensor.size(0)
        return F.log_softmax(cls_tensor.permute(0,2,3,1).contiguous()).view(n,-1,2), \
            box_tensor.permute(0,2,3,1).contiguous().view(n,-1,4)

    def forward(self, feature3, feature4, feature5) :
        # p3_cls, p3_box = self.permute(*self.det3(feature3))
        # p4_cls, p4_box = self.permute(*self.det4(feature4))
        # p5_cls, p5_box = self.permute(*self.det5(feature5))
        p3_cls, p3_box = self.permute(*self.det_module(feature3))
        p4_cls, p4_box = self.permute(*self.det_module(feature4))
        p5_cls, p5_box = self.permute(*self.det_module(feature5))
        classifications = torch.cat((p3_cls, p4_cls, p5_cls),dim=1)
        bbox_regression = torch.cat((p3_box, p4_box, p5_box),dim=1)
        return classifications, bbox_regression

class SSHLandmarkDetectionHead(nn.Module) :
    n_pyramid = 3
    __constants__ = ['n_landmarks', 'n_anchors']
    def __init__(self, in_channels : Union[int,List[int]], n_landmarks : int=5, n_anchors=2) :
        ## TODO : support extra pyramid
        super(SSHLandmarkDetectionHead, self).__init__()
        self.register_buffer('n_landmarks',torch.tensor([n_landmarks]).long())
        self.register_buffer('n_anchors',torch.tensor([n_anchors]).long())
        if isinstance(in_channels, int) :
            in_channels = [in_channels] * self.n_pyramid
        assert len(in_channels) == self.n_pyramid
        # self.det3 = SSHLandmarkDetectionModule(in_channels[0])
        # self.det4 = SSHLandmarkDetectionModule(in_channels[1])
        # self.det5 = SSHLandmarkDetectionModule(in_channels[2])
        out_channels = [2,4,n_landmarks*2]
        out_channels = [ch * n_anchors for ch in out_channels]

        assert all([in_channels[0]==ch for ch in in_channels])
        self.det_module = SSHLandmarkDetectionModule(in_channels[0],out_channels=out_channels)

    def permute(self, cls_tensor, box_tensor, lmk_tensor) :
        n = cls_tensor.size(0)
        ## temporary workaround for exporting since F.log_softmax failed, https://github.com/pytorch/pytorch/issues/20643
        ## log_softmax (and softmax) properly translated to onnx::LogSoftmax only when dim=-1, atleast for torch 1.4.0
        ## workaround : before log_softmax, transpose such that dim=-1
        ## TODO : consider removing log softmax since decoder also use softmax
        return F.log_softmax(cls_tensor.permute(0,1,3,2).contiguous(),dim=-1).permute(0,3,2,1).contiguous().view(n,-1,2), \
            box_tensor.permute(0,2,3,1).contiguous().view(n,-1,4), \
            lmk_tensor.permute(0,2,3,1).contiguous().view(n,-1,self.n_landmarks.item()*2)

    def forward(self, feature3, feature4, feature5) :
        # p3_cls, p3_box, p3_lmk = self.permute(*self.det3(feature3))
        # p4_cls, p4_box, p4_lmk = self.permute(*self.det4(feature4))
        # p5_cls, p5_box, p5_lmk = self.permute(*self.det5(feature5))
        p3_cls, p3_box, p3_lmk = self.permute(*self.det_module(feature3))
        p4_cls, p4_box, p4_lmk = self.permute(*self.det_module(feature4))
        p5_cls, p5_box, p5_lmk = self.permute(*self.det_module(feature5))
        classifications = torch.cat((p3_cls, p4_cls, p5_cls),dim=1)
        bbox_regression = torch.cat((p3_box, p4_box, p5_box),dim=1)
        lmks_regression = torch.cat((p3_lmk, p4_lmk, p5_lmk),dim=1)
        if self.training :
            output = bbox_regression, classifications, lmks_regression
        else :
            output = torch.cat((bbox_regression, classifications, lmks_regression), -1)
        return output