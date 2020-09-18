import torch
import torch.nn as nn
import torchvision.ops as ops
from typing import List, Tuple


class NoNMS:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, detections: torch.Tensor, class_indexes: torch.Tensor, bboxes: torch.Tensor, scores: torch.Tensor, iou_threshold: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        improvised from torch retinaface
        TODO : consider more extensible version, check dimension
        """
        if not len(scores.shape) == 1:
            raise RuntimeError(
                "expects `scores` to be 1-dimensional tensor, got %s with shape of %s" % (len(scores.shape), scores.shape))
        if not len(bboxes.shape) == 2:
            raise RuntimeError(
                "expects `bboxes` to be 2-dimensional tensor, got %s with shape of %s" % (len(bboxes.shape), bboxes.shape))
        if not len(detections.shape) == 3:
            raise RuntimeError("expects `detections` to be 3-dimensional tensor, got %s with shape of %s" %
                               (len(detections.shape), detections.shape))
        if detections.size()[0] > 1:
            raise RuntimeError("current version only support single batch")
        return detections, bboxes, scores, class_indexes


class NMS(nn.Module):
    def __init__(self, *args, **kwargs):
        super(type(self),self).__init__()
        pass

    def forward(self, detections: torch.Tensor, bboxes: torch.Tensor, scores: torch.Tensor, iou_threshold: torch.Tensor) -> torch.Tensor:
        """
        improvised from torch retinaface
        TODO : consider more extensible version, check dimension
        """
        if not len(scores.shape) == 1:
            raise RuntimeError(
                "expects `scores` to be 1-dimensional tensor, got %s with shape of %s" % (len(scores.shape), scores.shape))
        if not len(bboxes.shape) == 2:
            raise RuntimeError(
                "expects `bboxes` to be 2-dimensional tensor, got %s with shape of %s" % (len(bboxes.shape), bboxes.shape))
        if not len(detections.shape) == 3:
            raise RuntimeError("expects `detections` to be 3-dimensional tensor, got %s with shape of %s" %
                               (len(detections.shape), detections.shape))
        if detections.size()[0] > 1:
            raise RuntimeError("current version only support single batch")
        keep = ops.nms(bboxes, scores, iou_threshold)
        detections = detections.index_select(1, keep)
        return detections


def batched_nms(boxes, scores, idxs, iou_threshold):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor, float) -> torch.Tensor
    """
    adapted from torchvision.ops.boxes.batched_nms
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Parameters
    ----------
    boxes : Tensor[N, 4]
        boxes where NMS will be performed. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    idxs : Tensor[N]
        indices of the categories for each one of the boxes.
    iou_threshold : float
        discards all overlapping boxes
        with IoU > iou_threshold

    Returns
    -------
    keep : Tensor
        int64 tensor with the indices of
        the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    # offsets = _get_offsets(idxs, boxes)
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    ## workaround for onnx exporting, note maybe need to fix when input size > 2048
    max_coordinate = 2048 
    offsets = idxs * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = ops.nms(boxes_for_nms, scores, iou_threshold)
    return keep

class BatchedNMS(nn.Module):
    def __init__(self, *args, **kwargs):
        super(type(self),self).__init__()
        self.nms_fn = batched_nms

    def forward(self, detections: torch.Tensor, class_indexes: torch.Tensor, bboxes: torch.Tensor, scores: torch.Tensor, iou_threshold: torch.Tensor) -> torch.Tensor:
        if not len(class_indexes.shape) == 1:
            raise RuntimeError("expects `class_indices` to be 1-dimensional tensor, got %s with shape of %s" %
                               (len(class_indexes.shape), class_indexes.shape))
        if not len(scores.shape) == 1:
            raise RuntimeError(
                "expects `scores` to be 1-dimensional tensor, got %s with shape of %s" % (len(scores.shape), scores.shape))
        if not len(bboxes.shape) == 2:
            raise RuntimeError(
                "expects `bboxes` to be 2-dimensional tensor, got %s with shape of %s" % (len(bboxes.shape), bboxes.shape))
        if not len(detections.shape) == 3:
            raise RuntimeError("expects `detections` to be 3-dimensional tensor, got %s with shape of %s" %
                               (len(detections.shape), detections.shape))
        if detections.size()[0] > 1:
            raise RuntimeError("current version only support single batch")
        keep = self.nms_fn(
            bboxes, scores, class_indexes, iou_threshold)
        detections = detections.index_select(1, keep)
        return detections
