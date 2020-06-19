import torch
import torch.nn as nn

from torch import Tensor
from typing import Tuple, Union        

def permute(features : Tensor, n_elements : int) :
    ## actually permute_and_reshape
    n = features.size(0)
    return features.permute(0,2,3,1).contiguous().view(n, -1, n_elements)

class RetinaHead(nn.Module) :
    __constants__ = ['n_classes']
    def __init__(self, in_channels : int, n_anchors : int, n_classes : int, n_conv : int=4) :
        super(RetinaHead, self).__init__()
        bbox_layers, cls_layers = [], []
        for _ in range(n_conv) :
            bbox_layers.append(nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,padding=1))
            bbox_layers.append(nn.ReLU(True))
            cls_layers.append(nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,padding=1))
            cls_layers.append(nn.ReLU(True))
        self.bbox_subnet = nn.Sequential(*bbox_layers, nn.Conv2d(in_channels, n_anchors*4,kernel_size=3,padding=1))
        self.cls_subnet = nn.Sequential(*cls_layers, nn.Conv2d(in_channels, n_anchors*n_classes,kernel_size=3,padding=1))
        self.n_classes = n_classes
        self.n_anchors = n_anchors
    
    def forward(self, p3 : Tensor, p4 : Tensor, p5 : Tensor, extra : Union[tuple,Tensor,list]=None) -> Tensor:
        ## TODO : support extra pyramid
        n = p3.size(0)
        box3 = permute(self.bbox_subnet(p3),4)
        box4 = permute(self.bbox_subnet(p4),4)
        box5 = permute(self.bbox_subnet(p5),4)
        cls3 = permute(self.cls_subnet(p3),self.n_classes)
        cls4 = permute(self.cls_subnet(p4),self.n_classes)
        cls5 = permute(self.cls_subnet(p5),self.n_classes)
        extra_box, extra_cls = [], []
        ## TODO : make torch.script-able
        if not extra is None :
            if isinstance(extra, Tensor) :
                extra = [extra]
            for ext in extra :
                extra_box.append(permute(self.bbox_subnet(ext),4))
                extra_cls.append(permute(self.cls_subnet(ext),self.n_classes))
        bbox_pred = torch.cat((box3, box4, box5, *extra_box), dim=1)
        cls_pred = torch.cat((cls3, cls4, cls5, *extra_cls), dim=1)
        if self.training :
            output = bbox_pred, cls_pred
        else :
            output = torch.cat((bbox_pred, cls_pred), -1)
        return output