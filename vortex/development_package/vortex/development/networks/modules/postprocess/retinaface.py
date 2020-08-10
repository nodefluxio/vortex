import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor, Size, tensor
from typing import Tuple, List, Union

from ..losses.utils.ssd import decode_landm, decode

class RetinaFaceDecoder(nn.Module) :
    __constants__ = ['variance', 'priors', 'n_landmarks']
    def __init__(self, priors : Tensor, variance : Tensor, n_landmarks : int) :
        super(RetinaFaceDecoder, self).__init__()
        self.register_buffer('variance', variance)
        self.register_buffer('priors', priors)
        self.register_buffer('n_landmarks', tensor([n_landmarks]).long())
    
    def forward(self, input : Tensor, score_threshold : Tensor) -> Tuple[Tensor,Tensor,Tensor,Tensor] :
        assert input.dim() == 3
        assert input.size(0) == 1
        assert input.size(2) == (6 + self.n_landmarks*2)
        assert score_threshold.dim() == 1
        assert score_threshold.size() == Size([1])
        bounding_boxes, classifications, landmarks = input[...,0:4], input[...,4:6], input[...,6:]
        classifications = F.softmax(classifications,dim=-1).squeeze(0) ## squeeze(0) assuming single batch, 1 means fg
        class_conf, class_pred = classifications.max(1, keepdim=True)
        object_indices = torch.gt(class_pred, 0) ## take object only
        object_indices = object_indices.squeeze(1).nonzero().squeeze(1)
        class_conf = class_conf.index_select(0,object_indices)
        class_pred = class_pred.index_select(0,object_indices)
        bounding_boxes = decode(bounding_boxes.squeeze(0), self.priors, variances=self.variance)
        landmarks = decode_landm(landmarks.squeeze(0), self.priors, variances=self.variance)
        bounding_boxes = bounding_boxes.index_select(0,object_indices)
        landmarks = landmarks.index_select(0,object_indices)
        indices = torch.gt(class_conf, score_threshold)
        indices = indices.squeeze(1).nonzero().squeeze(1)
        bounding_boxes = bounding_boxes.index_select(0, indices)
        landmarks = landmarks.index_select(0, indices)
        class_conf = class_conf.index_select(0, indices)
        class_pred = class_pred.index_select(0, indices)
        class_pred = class_pred - 1
        detections = torch.cat((bounding_boxes,class_conf,class_pred.float(),landmarks),dim=-1)
        return bounding_boxes, class_conf.squeeze(1), class_pred.squeeze(1), detections.unsqueeze(0)
    
from .base_postprocess import BasicNMSPostProcess, BatchedNMSPostProcess

class RetinaFacePostProcess(BatchedNMSPostProcess) :
    """ Post-Process for retinaface, comply with basic detector post process
    """
    def __init__(self, priors : Tensor, variance : Tensor, n_landmarks : int, *args, **kwargs):
        super(RetinaFacePostProcess, self).__init__(
            decoder=RetinaFaceDecoder(priors=priors, variance=variance, n_landmarks=n_landmarks),
            *args, **kwargs
        )

def get_postprocess(*args, **kwargs) :
    return RetinaFacePostProcess(*args,**kwargs)