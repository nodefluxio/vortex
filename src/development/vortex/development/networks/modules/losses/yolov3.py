import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from typing import Union, List, Dict, Tuple

from .utils.yolov3 import build_targets

SUPPORTED_METHODS = [
    'YoloV3Loss'
]


class _DetectorWeightedLoss(nn.Module):
    """improvised from nn.modules.loss._WeightedLoss"""

    def __init__(self, weight_fg, weight_bg, ignore_thresh, weight_classes, weight_conf, weight_loc, reduction='mean'):
        super(_DetectorWeightedLoss, self).__init__()
        self.reduction = reduction
        self.register_buffer('weight_fg', weight_fg)
        self.register_buffer('weight_bg', weight_bg)
        self.register_buffer('weight_classes', weight_classes)
        self.register_buffer('weight_conf', weight_conf)
        self.register_buffer('weight_loc', weight_loc)
        self.register_buffer('ignore_thresh', ignore_thresh)


class YoloV3Loss(_DetectorWeightedLoss):
    __constants__ = ['weight_fg', 'weight_bg', 'reduction', 'ignore_thresh',
                     'weight_loc', 'weight_classes', 'weight_conf', 'anchors']

    def __init__(self, weight_fg: float, weight_bg: float, ignore_thresh: float, weight_loc: float = 1.0, weight_classes: float = 1.0, weight_conf: float = 1.0, reduction='mean', check=True):
        super(YoloV3Loss, self).__init__(
            torch.Tensor([weight_fg]),
            torch.Tensor([weight_bg]),
            torch.Tensor([ignore_thresh]),
            torch.Tensor([weight_classes]),
            torch.Tensor([weight_conf]),
            torch.Tensor([weight_loc]),
            reduction
        )
        self.check = check

    def assign_anchors(self, anchors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        self.register_buffer('anchors', torch.stack(anchors))

    def compute_loss(self, det: torch.Tensor, targets: torch.Tensor, anchors: torch.Tensor, device):
        x = det[..., 0]
        y = det[..., 1]
        w = det[..., 2]
        h = det[..., 3]
        pred_boxes = det[..., 0:4]
        pred_conf = det[..., 4]
        pred_cls = det[..., 5:]
        # prevent silent error
        if not (torch.all(x >= 0.) and torch.all(x <= 1.) and torch.all(y >= 0.) and torch.all(y <= 1.)):
            if self.check:
                x_min, x_max = x[x < 0.], x[x > 1.]
                y_min, y_max = y[y < 0.], y[y > 1.]
                raise RuntimeError("YOLO Loss assume xy center is [0.,1.], got %s" % (
                    [x_min, x_max, y_min, y_max]))
            else:
                warnings.warn("YOLO Loss assume xy center is [0.,1.]")
        targets = build_targets(
            pred_shape=det.shape[0:3],
            pred_cls=pred_cls,
            targets=targets,
            anchors=anchors,
            ignore_thresh=self.ignore_thresh,
            device=device
        )
        class_mask, fg_mask, bg_mask, tx, ty, tw, th, tc, tconf = targets
        return components_loss(
            x=x[fg_mask], tx=tx[fg_mask],
            y=y[fg_mask], ty=ty[fg_mask],
            w=w[fg_mask], tw=tw[fg_mask],
            h=h[fg_mask], th=th[fg_mask],
            conf_fg=pred_conf[fg_mask], tconf_fg=tconf[fg_mask],
            conf_bg=pred_conf[bg_mask], tconf_bg=tconf[bg_mask],
            classes=pred_cls[fg_mask], tclasses=tc[fg_mask],
            weight_fg=self.weight_fg,
            weight_bg=self.weight_bg,
            weight_classes=self.weight_classes,
            weight_conf=self.weight_conf,
            weight_loc=self.weight_loc,
            reduction=self.reduction
        )

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """
        compute yolo loss
        """
        device = self.weight_fg.device
        anchors = self.anchors
        if anchors is None:
            raise RuntimeError("please assign anchors before computing loss")
        losses = sum([self.compute_loss(prediction, targets, anchor, device)
                      for prediction, anchor in zip(input, anchors)])
        return losses


def components_loss(
    x: torch.Tensor, y: torch.Tensor,
    w: torch.Tensor, h: torch.Tensor,
    tx: torch.Tensor, ty: torch.Tensor,
    tw: torch.Tensor, th: torch.Tensor,
    classes: torch.Tensor, tclasses: torch.Tensor,
    conf_fg: torch.Tensor,  conf_bg: torch.Tensor,
    tconf_fg: torch.Tensor, tconf_bg: torch.Tensor,
    weight_fg: float, weight_bg: float, reduction: str,
    weight_loc: float = 1.0, weight_classes: float = 1.0, weight_conf: float = 1.0
):
    """
    compute yolo loss in functional fashion, 
    adapted from :
        https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/47b7c912877ca69db35b8af3a38d6522681b3bb3/models.py#L191
    TODO : consider to reduce number of args
    """

    loss_x = F.mse_loss(x, tx, reduction=reduction)
    loss_y = F.mse_loss(y, ty, reduction=reduction)
    loss_w = F.mse_loss(w, tw, reduction=reduction)
    loss_h = F.mse_loss(h, th, reduction=reduction)
    loss_loc = loss_x + loss_y + loss_w + loss_h
    loss_fg = F.binary_cross_entropy(conf_fg, tconf_fg, reduction=reduction)
    loss_bg = F.binary_cross_entropy(conf_bg, tconf_bg, reduction=reduction)
    loss_conf = weight_bg * loss_bg + weight_fg * loss_fg
    loss_classes = F.binary_cross_entropy(
        classes, tclasses, reduction=reduction)
    return loss_loc * weight_loc + loss_conf * weight_conf + loss_classes * weight_classes


# def create_loss(*args, **kwargs):
#     return YoloV3Loss(*args, **kwargs)
