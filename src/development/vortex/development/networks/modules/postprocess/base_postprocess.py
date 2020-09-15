import torch
import torchvision
import torch.nn as nn
from typing import Type, List, Tuple, Dict, Callable, Union


def check_annotations(lhs, rhs):
    lhs_args = [lhs]
    rhs_args = [rhs]
    if lhs.__class__ == type(Union):
        lhs_args = list(lhs.__args__)
    if rhs.__class__ == type(Union):
        rhs_args = list(rhs.__args__)
    return any([arg in rhs_args for arg in lhs_args])


class BasicNMSPostProcess(nn.Module):
    """ 
    Standardized detector post-process; takes Callable decoder and Callable nms;
    The following forward operation are defined :
    ```
        bboxes, scores, class_indexes, detections = decoder(
            input, score_threshold
        )
        results = nms(
            detections, bboxes,
            scores, iou_threshold
        )
    ```
    decoder and nms signatures are enforced with the following signature :
    ```
        decoder (input : Tensor) -> Tuple[Tensor,Tensor,Tensor,Tensor]
        nms (bboxes : Tensor, class_indexes : Tensor, scores : Tensor, iou_threshold : Tensor) -> Union[Tensor,Tuple[Tensor,Tensor]]
    ```
    """
    decoder_signature = {
        'input': torch.Tensor,
        'return': Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    }
    nms_signature = {
        'bboxes': torch.Tensor,
        'scores': torch.Tensor,
        'class_indexes': torch.Tensor,
        'iou_threshold': torch.Tensor,  # float,
        'return': Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    }

    def __init__(self, decoder: Callable, nms: bool = True):
        from .utils.nms import BatchedNMS, NoNMS
        super(BasicNMSPostProcess, self).__init__()
        self.decoder = decoder
        self.nms = BatchedNMS() if nms else NoNMS()

        # check for decoder annotations
        try:
            annotations = self.decoder.forward.__annotations__
        except AttributeError as e:
            try:
                annotations = self.decoder.__annotations__
            except AttributeError as e:
                raise RuntimeError("please annotate decoder or decoder.forward with %s" %
                                   BasicNMSPostProcess.decoder_signature.keys())
        if not len(annotations):
            raise RuntimeError("please annotate decoder with %s" %
                               BasicNMSPostProcess.decoder_signature.keys())
        # check for decoder type signature
        for key, value in BasicNMSPostProcess.decoder_signature.items():
            if not key in annotations.keys():
                raise TypeError("missing decoder annotation(s) : %s" % key)
            if value != annotations[key]:
                raise TypeError("decoder type mismatch for %s, expects %s got %s" % (
                    key, value, annotations[key]))

        # check for nms annotations
        try:
            annotations = self.nms.__annotations__
        except AttributeError as e:
            try:
                annotations = self.nms.forward.__annotations__
            except AttributeError as e:
                try:
                    annotations = self.nms.__call__.__annotations__
                except AttributeError as e:
                    raise TypeError("please annotate nms with %s" %
                                    BasicNMSPostProcess.nms_signature.keys())
        if not len(annotations):
            raise TypeError("please annotate nms with %s" %
                            BasicNMSPostProcess.nms_signature.keys())
        # check for decoder type signature
        for key, value in BasicNMSPostProcess.nms_signature.items():
            if not key in annotations.keys():
                raise TypeError("missing nms annotation(s) : %s" % key)
            if not check_annotations(value, annotations[key]):
                raise TypeError("nms type mismatch for %s, expects %s got %s" % (
                    key, value, annotations[key]))
        self.additional_inputs = (
            ('score_threshold', (1,)),
            ('iou_threshold', (1,)),
        )

    def _forward(self, input: torch.Tensor, score_threshold: torch.Tensor, iou_threshold: torch.Tensor) -> torch.Tensor:
        bboxes, scores, class_indexes, detections = self.decoder(
            input=input,
            score_threshold=score_threshold
        )
        results = self.nms(
            detections=detections,
            bboxes=bboxes,
            scores=scores,
            class_indexes=class_indexes,
            iou_threshold=iou_threshold
        )
        return results
    
    def forward(self, input: torch.Tensor, score_threshold: torch.Tensor, iou_threshold: torch.Tensor) -> torch.Tensor:
        return self._forward(input, score_threshold, iou_threshold)

class BatchedNMSPostProcess(BasicNMSPostProcess) :
    def __init__(self, decoder: Callable, nms: bool = True) :
        super(BatchedNMSPostProcess,self).__init__(decoder, nms)

    def forward(self, input: torch.Tensor, score_threshold: torch.Tensor, iou_threshold: torch.Tensor) :
        n_batch = input.size(0)
        results = []
        for i in range(n_batch) :
            results.append(self._forward(
                input[i].unsqueeze(0), score_threshold, iou_threshold
            ).squeeze(0))
        return tuple(results)

class SoftmaxPostProcess(nn.Module):
    def __init__(self, dim=1, keepdim=False):
        super(SoftmaxPostProcess, self).__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = self.softmax(input)
        conf_label, cls_label = input.max(dim=self.dim, keepdim=self.keepdim)
        return torch.stack((cls_label.float(), conf_label), dim=1)


class NoOpPostProcess(nn.Module):
    def __init__(self):
        super(NoOpPostProcess, self).__init__()

    def forward(self, input, *args, **kwargs):
        return input
