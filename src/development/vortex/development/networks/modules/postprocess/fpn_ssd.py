import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor, Size, tensor
from typing import Tuple, List, Union

from ..losses.utils.ssd import decode_landm, decode

class FPNSSDDecoder(nn.Module) :
    __constants__ = ['variance', 'priors', 'n_classes']
    def __init__(self, priors : Tensor, variance : Tensor, n_classes : int) :
        super(FPNSSDDecoder, self).__init__()
        self.register_buffer('variance', variance)
        self.register_buffer('priors', priors)
        self.register_buffer('n_classes', tensor([n_classes]).long())
    
    def forward(self, input : Tensor, score_threshold : Tensor) -> Tuple[Tensor,Tensor,Tensor,Tensor] :
        assert input.dim() == 3
        assert input.size(0) == 1   ## single batch for now
        assert input.size(2) == (4 + self.n_classes)
        assert score_threshold.dim() == 1
        assert score_threshold.size() == Size([1])
        bounding_boxes = input[...,0:4]
        classifications = input[...,4:]
        classifications = F.softmax(classifications,dim=-1).squeeze(0)
        class_conf, class_pred = classifications.max(1, keepdim=True)
        ## take object only
        object_indices = torch.gt(class_pred, 0)
        object_indices = object_indices.squeeze(1).nonzero().squeeze(1)
        class_conf = class_conf.index_select(0,object_indices)
        class_pred = class_pred.index_select(0,object_indices)
        bounding_boxes = decode(bounding_boxes.squeeze(0), self.priors, variances=self.variance)
        bounding_boxes = bounding_boxes.index_select(0,object_indices)
        ## take conf greater than threshold
        indices = torch.gt(class_conf, score_threshold)
        indices = indices.squeeze(1).nonzero().squeeze(1)
        bounding_boxes = bounding_boxes.index_select(0,indices)
        class_conf = class_conf.index_select(0,indices)
        class_pred = class_pred.index_select(0,indices)
        class_pred = class_pred - 1
        ## final detection tensor
        detections = torch.cat((bounding_boxes,class_conf,class_pred.float()),dim=-1)
        return bounding_boxes, class_conf.squeeze(1), class_pred.squeeze(1), detections.unsqueeze(0)
    
from .base_postprocess import BatchedNMSPostProcess

class FPNSSDPostProcess(BatchedNMSPostProcess) :
    """ Post-Process for retinaface, comply with basic detector post process
    """
    def __init__(self, priors : Tensor, variance : Tensor, n_classes : int, *args, **kwargs):
        super(FPNSSDPostProcess, self).__init__(
            decoder=FPNSSDDecoder(priors=priors, variance=variance, n_classes=n_classes),
            *args, **kwargs
        )

def get_postprocess(*args, **kwargs) :
    return FPNSSDPostProcess(*args,**kwargs)