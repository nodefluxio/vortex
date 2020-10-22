import torch
import torch.nn as nn
from typing import Union, Tuple, List

from .base_postprocess import BatchedNMSPostProcess


def yolo2xywh(bboxes: torch.Tensor):
    if not (len(bboxes.size()) == 3):
        raise RuntimeError(
            "this routine expects predictions is a 3-dimensional tensor! got %s dimension" % len(bboxes.size()))
    x = bboxes[..., 0] - bboxes[..., 2] / 2
    y = bboxes[..., 1] - bboxes[..., 3] / 2
    w = bboxes[..., 0] + bboxes[..., 2] / 2
    h = bboxes[..., 1] + bboxes[..., 3] / 2
    return torch.stack((x, y, w, h), 2)


class YoloV3Decoder(nn.Module):
    def __init__(self, img_size, threshold: bool = True):
        # note : some backend do not have onnx `NonZero` op,
        # which is necessary for thresholding,
        # `threshold` param allows to turn it off
        super(YoloV3Decoder, self).__init__()
        self.threshold = threshold
        self.img_size = img_size

    def forward(self, input: torch.Tensor, score_threshold: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        improvised from torch retinaface
        TODO : consider more extensible version, check dimension
        """
        predictions = input
        if torch.any(torch.isnan(predictions)):
            raise ValueError("your predictions have nan! please check")
        if not (len(predictions.size()) == 3):
            raise RuntimeError(
                "this routine expects predictions is a 3-dimensional tensor! got %s dimension" % len(predictions.size()))
        # yolo darknet format : cx cy w h is_obj class...
        class_conf, class_pred = predictions[..., 5:].max(2, keepdim=True)
        # check if we should perform thresholding
        if self.threshold:
            indices = (predictions[..., 4] > score_threshold)
            indices = indices.squeeze(0)
            indices = torch.nonzero(indices)
            predictions = predictions.index_select(1, indices.squeeze(1))
            class_conf = class_conf.index_select(1, indices.squeeze(1))
            class_pred = class_pred.index_select(1, indices.squeeze(1))
            # altervative for thresholding
            # indices = (predictions[...,4] * class_conf.squeeze(2) > score_threshold)
            # indices = indices.squeeze(0)
            # indices = torch.nonzero(indices)
            # predictions = predictions.index_select(1, indices.squeeze(1))
            # class_conf = class_conf.index_select(1, indices.squeeze(1))
            # class_pred = class_pred.index_select(1, indices.squeeze(1))
            # indices = (class_conf.squeeze(2) > score_threshold)
            # indices = indices.squeeze(0)
            # indices = torch.nonzero(indices)
            # predictions = predictions.index_select(1, indices.squeeze(1))
            # class_conf = class_conf.index_select(1, indices.squeeze(1))
            # class_pred = class_pred.index_select(1, indices.squeeze(1))
        bboxes = predictions[..., :4] / self.img_size
        objectness = predictions[..., 4]
        scores = objectness * class_conf.squeeze(2)
        bboxes = yolo2xywh(bboxes)
        detections = torch.cat((bboxes, class_conf, class_pred.float()), 2)
        return bboxes.squeeze(0), scores.squeeze(0), class_pred.squeeze(0).squeeze(1), detections


class YoloV3PostProcess(BatchedNMSPostProcess):
    """ Post-Process for yolo, comply with basic detector post process
    """

    def __init__(self, img_size, threshold: bool = True, *args, **kwargs):
        super(YoloV3PostProcess, self).__init__(
            decoder=YoloV3Decoder(img_size=img_size, threshold=threshold),
            *args, **kwargs
        )


def get_postprocess(*args, **kwargs):
    """ return postprocess method for yolo
    """
    return YoloV3PostProcess(*args, **kwargs)
