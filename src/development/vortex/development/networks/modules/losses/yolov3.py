import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from typing import Tuple

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

    def __init__(self, weight_fg: float, weight_bg: float, ignore_thresh: float, weight_loc: float = 1.0, 
                 weight_classes: float = 1.0, weight_conf: float = 1.0, reduction='mean', check=True):
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
        x, y = det[..., 0], det[..., 1]
        w, h = det[..., 2], det[..., 3]
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


class YoloV3LossDena(nn.Module):
    def __init__(self, img_size, anchors, ref_anchors, mask_anchors, ignore_thresh=0.7, 
                 weight_loc=1.0, weight_classes=1.0, weight_conf=1.0):
        super(YoloV3LossDena, self).__init__()
        self.img_size = img_size
        self.ignore_thresh = ignore_thresh
        self.weight_loc = weight_loc
        self.weight_classes = weight_classes
        self.weight_conf = weight_conf

        num_anchor, num_scale = len(anchors), len(anchors[0])
        self.register_buffer('anchors', torch.stack(anchors))
        self.register_buffer('ref_anchors', torch.stack(ref_anchors))
        self.mask_anchors = mask_anchors

        anchor_wh = [anc.view(1, num_anchor, 1, 1, 2) for anc in self.anchors.clone()]
        self.register_buffer('anchors_wh', torch.stack(anchor_wh))
        self._gs = [0]*num_anchor
        for n, gs in enumerate([img_size//(2**5), img_size//(2**4), img_size//(2**3)]):
            self.create_grid(n, gs, gs)

    def create_grid(self, n, gx, gy):
        grid_y, grid_x = torch.meshgrid(torch.arange(gy), torch.arange(gx))
        grid_xy = torch.stack((grid_x, grid_y), dim=2).view(1, 1, gy, gx, 2)
        grid_xy = grid_xy.type(torch.float32).to(self.anchors_wh.device)
        self.register_buffer("grid_xy_{}".format(n), grid_xy)
        self._gs[n] = (gx, gy)

    def compute_loss_layer(self, pred, targets, n_layer, num_targets):
        device = pred.device
        anchor = self.anchors[n_layer]
        mask_anchor = self.mask_anchors[n_layer]
        ref_anchor = self.ref_anchors[n_layer]
        nx, ny = pred.shape[2:4]
        n_classes = pred.shape[4] - 5

        if self._gs[n_layer][0] != nx or self._gs[n_layer][1] != ny:
            self.create_grid(n_layer, nx, ny)
        grid_xy = getattr(self, "grid_xy_{}".format(n_layer))
        anchor_wh = self.anchors_wh[n_layer]

        pred_bbox = pred[..., :4].clone()
        pred_bbox[..., 0:2] += grid_xy
        pred_bbox[..., 2:4] = torch.exp(pred_bbox[..., 2:4]) * anchor_wh

        labels = targets.clone()
        labels[..., 1::2] *= nx
        labels[..., 2::2] *= ny

        tgt_mask = torch.zeros(*pred.shape[:4], 4 + n_classes, 
            dtype=torch.float32, device=device)
        tgt_scale = torch.zeros(*pred.shape[:4], 2, 
            dtype=torch.float32, device=device)
        obj_mask = torch.ones(*pred.shape[:4], 
            dtype=torch.float32, device=device)
        tgt_encoded = torch.zeros_like(pred)
        for b, n in enumerate(num_targets):
            if n == 0:
                continue
            label_bbox = torch.zeros(n, 4, dtype=torch.float32, device=device)
            label_bbox[:n, 2:4] = labels[b, :n, 3:5].clone()

            anchor_ious = bboxes_iou(label_bbox, ref_anchor)
            best_anchor_idx = anchor_ious.argmax(dim=1)
            best_anchor_mask = ((best_anchor_idx == mask_anchor[0]) | 
                (best_anchor_idx == mask_anchor[1]) | (best_anchor_idx == mask_anchor[2]))
            best_anchor = (best_anchor_idx % 3)
            if sum(best_anchor_mask) == 0:
                continue

            label_bbox[:n, 0:2] = labels[b, :n, 1:3].clone()
            pred_ious = bboxes_iou(pred_bbox[b].view(-1, 4), label_bbox, xyxy=False)
            pred_best_iou,_ = pred_ious.max(dim=1)
            pred_best_iou = (pred_best_iou > self.ignore_thresh).view(pred_bbox[b].shape[:3])
            obj_mask[b] = torch.logical_not(pred_best_iou)

            idxs = (best_anchor_mask == True).nonzero().flatten().tolist()
            label_int_x = labels[b, :n, 1].to(torch.int).tolist()
            label_int_y = labels[b, :n, 2].to(torch.int).tolist()
            for idx in idxs:
                i,j = label_int_x[idx], label_int_y[idx]
                a = best_anchor[idx].item()
                obj_mask[b, a, j, i] = 1
                tgt_mask[b, a, j, i, :] = 1
                tgt_scale[b, a, j, i, :] = torch.sqrt(
                    2 - (labels[b, idx, 3] * labels[b, idx, 4] / nx / ny))
                tgt_encoded[b, a, j, i, 0:2] = labels[b, idx, 1:3] - \
                    labels[b, idx, 1:3].type(torch.int).type(torch.float)
                tgt_encoded[b, a, j, i, 2:4] = torch.log(
                    labels[b, idx, 3:5] / anchor[a] + 1e-16)
                tgt_encoded[b, a, j, i, 4] = 1
                tgt_encoded[b, a, j, i, labels[b, idx, 0].type(torch.int) + 5] = 1

        pred[..., 4] *= obj_mask
        pred[..., [*range(0, 4), *range(5, pred.shape[4])]] *= tgt_mask
        pred[..., 2:4] *= tgt_scale
        tgt_encoded[..., 4] *= obj_mask
        tgt_encoded[..., [*range(0, 4), *range(5, pred.shape[4])]] *= tgt_mask
        tgt_encoded[..., 2:4] *= tgt_scale

        loss_xy = F.binary_cross_entropy(pred[..., 0:2], tgt_encoded[..., 0:2], 
            weight=tgt_scale*tgt_scale, reduction="sum")
        loss_wh = F.mse_loss(pred[..., 2:4], tgt_encoded[..., 2:4], reduction="sum") / 2
        loss_obj = F.binary_cross_entropy(pred[..., 4], tgt_encoded[..., 4], reduction="sum")
        loss_cls = F.binary_cross_entropy(pred[..., 5:], tgt_encoded[..., 5:], reduction="sum")
        loss = self.weight_loc * (loss_xy + loss_wh) + self.weight_conf * loss_obj + \
            self.weight_classes * loss_cls
        assert not torch.isnan(loss).item(), "Loss is Nan at {}\n{} {} {} {}".format(n_layer, loss_xy, loss_wh, loss_obj, loss_cls)
        return loss

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        num_targets = (targets.sum(dim=2) > 0).sum(dim=1).tolist()
        if self.anchors is None:
            raise RuntimeError("please assign anchors before computing loss")
        losses = sum([self.compute_loss_layer(pred, targets, n, num_targets)
                      for n, pred in enumerate(input)])
        return losses

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    # top left
    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                        (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                        (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


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
