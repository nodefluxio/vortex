import torch
import torch.nn as nn

from torch import Tensor
from typing import Tuple, Union, List  

def permute(features : Tensor, n_elements : int) :
    ## actually permute_and_reshape
    n = features.size(0)
    return features.permute(0,2,3,1).contiguous().view(n, -1, n_elements)

class SSDHead(nn.Module) :
    __constants__ = ['n_classes', 'n_anchors', 'n_extras']
    def __init__(self, backbone_channels : Union[list,int], extra_channels : Union[int,list], n_anchors : int, n_classes : int, n_conv : int=4) :
        super(SSDHead, self).__init__()
        if isinstance(backbone_channels, int) :
            backbone_channels = [backbone_channels] * 3
        assert len(backbone_channels) == 3
        self.bbox_subnet, self.cls_subnet = nn.ModuleList(), nn.ModuleList()
        for i, in_channels in enumerate(backbone_channels) :
            bbox_layers, cls_layers = [], []
            for _ in range(n_conv) :
                bbox_layers.append(nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,padding=1))
                bbox_layers.append(nn.ReLU(True))
                cls_layers.append(nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,padding=1))
                cls_layers.append(nn.ReLU(True))
            bbox_subnet = nn.Sequential(*bbox_layers, nn.Conv2d(in_channels, n_anchors*4,kernel_size=3,padding=1))
            cls_subnet = nn.Sequential(*cls_layers, nn.Conv2d(in_channels, n_anchors*n_classes,kernel_size=3,padding=1))
            self.bbox_subnet.append(bbox_subnet)
            self.cls_subnet.append(cls_subnet)
        n_extras = len(extra_channels)
        extra_bbox_subnet, extra_cls_subnet = nn.ModuleList(), nn.ModuleList()
        for i, in_channels in enumerate(extra_channels) :
            bbox_layers, cls_layers = [], []
            for _ in range(n_conv) :
                bbox_layers.append(nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,padding=1))
                bbox_layers.append(nn.ReLU(True))
                cls_layers.append(nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,padding=1))
                cls_layers.append(nn.ReLU(True))
            bbox_subnet = nn.Sequential(*bbox_layers, nn.Conv2d(in_channels, n_anchors*4,kernel_size=3,padding=1))
            cls_subnet = nn.Sequential(*cls_layers, nn.Conv2d(in_channels, n_anchors*n_classes,kernel_size=3,padding=1))
            extra_bbox_subnet.append(bbox_subnet)
            extra_cls_subnet.append(cls_subnet)
        self.extra_bbox_subnet = extra_bbox_subnet
        self.extra_cls_subnet = extra_cls_subnet
        self.n_classes = n_classes
        self.n_anchors = n_anchors
        self.n_extras = n_extras
    
    def forward(self, p3 : Tensor, p4 : Tensor, p5 : Tensor, extra : Union[tuple,list]) -> Tensor:
        ## TODO : support extra pyramid
        n = p3.size(0)
        box3 = permute(self.bbox_subnet[0](p3),4)
        box4 = permute(self.bbox_subnet[1](p4),4)
        box5 = permute(self.bbox_subnet[2](p5),4)
        cls3 = permute(self.cls_subnet[0](p3),self.n_classes)
        cls4 = permute(self.cls_subnet[1](p4),self.n_classes)
        cls5 = permute(self.cls_subnet[2](p5),self.n_classes)
        extra_box, extra_cls = [], []
        ## TODO : make torch.script-able
        assert len(extra) == self.n_extras
        for i, ext in enumerate(extra) :
            extra_box.append(permute(self.extra_bbox_subnet[i](ext),4))
            extra_cls.append(permute(self.extra_cls_subnet[i](ext),self.n_classes))
        bbox_pred = torch.cat((box3, box4, box5, *extra_box), dim=1)
        cls_pred = torch.cat((cls3, cls4, cls5, *extra_cls), dim=1)
        if self.training :
            output = bbox_pred, cls_pred
        else :
            output = torch.cat((bbox_pred, cls_pred), -1)
        return output