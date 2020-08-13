from typing import Union, List, Dict, Tuple

import torch

__all__ = [
    'encode_grid_labels',
    'encode_yolo_bbox_labels',
    'build_targets',
]


def bbox_iou(box1, box2, x1y1x2y2=True):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:
        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                 (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    zero = 1e-16    # numerical stability for iou
    # Union Area
    union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + zero) + \
                 (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

    iou = inter_area / union_area  # iou

    return iou


def bbox_wh_iou(wh1, wh2):
    zero = 1e-16
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1*h1 + zero) + w2 * h2 - inter_area
    return inter_area / union_area


def wh_iou(box1, box2):
    # Returns the IoU of wh1 to wh2. wh1 is 2, wh2 is nx2
    box2 = box2.t()

    # w, h = box1
    w1, h1 = box1[0], box1[1]
    w2, h2 = box2[0], box2[1]

    # Intersection area
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)

    # Union Area
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area

    return inter_area / union_area  # iou


def encode_grid_labels(det_shape, targets, anchors, ignore_thresh, device: Union[str, torch.device] = 'cuda'):
    """
    encode label in relative image format to grid format
    """
    nB, nA, nG = det_shape
    is_obj_mask = torch.zeros(nB, nA, nG, nG, device=device, dtype=torch.bool)
    no_obj_mask = torch.ones(nB, nA, nG, nG, device=device, dtype=torch.bool)

    # TODO : consider using named Tensor on pytorch 1.3
    target_boxes = targets[:, 2:6] * nG
    target_boxes[:, :2] += target_boxes[:, 2:] / 2.
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    b, target_labels = targets[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    gi = torch.clamp(gi, 0, is_obj_mask.shape[3]-1)
    gj = torch.clamp(gj, 0, is_obj_mask.shape[2]-1)
    is_obj_mask[b, best_n, gj, gi] = 1
    no_obj_mask[b, best_n, gj, gi] = 0

    for i, anchor_ious in enumerate(ious.t()):
        no_obj_mask[b[i], anchor_ious > ignore_thresh, gj[i], gi[i]] = 0
    # TODO : return with nicer format
    return b, best_n, gx, gy, gw, gh, gi, gj, is_obj_mask, no_obj_mask, target_labels, target_boxes


def encode_yolo_bbox_labels(det_shape, n_classes, targets, anchors, ignore_thresh, device: Union[str, torch.device] = 'cuda'):
    """
    encode labels to grids with yolo format, see yolov3 chapter 2.1
    Reference :
    [1] J. Redmon and A. Farhadi, “YOLOv3: An Incremental Improvement,” Apr. 2018.
    """
    nB, nA, nG = det_shape
    # nC = det_shape[-1]
    nC = n_classes
    tx = torch.zeros(nB, nA, nG, nG, device=device)
    ty = torch.zeros(nB, nA, nG, nG, device=device)
    tw = torch.zeros(nB, nA, nG, nG, device=device)
    th = torch.zeros(nB, nA, nG, nG, device=device)
    tc = torch.zeros(nB, nA, nG, nG, nC, device=device)

    b, best_n, gx, gy, gw, gh, gi, gj, is_obj_mask, no_obj_mask, target_labels, target_boxes = encode_grid_labels(
        det_shape, targets, anchors, ignore_thresh, device)

    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    zero = 1e-16  # numerical stability for log
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + zero)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + zero)
    tc[b, best_n, gj, gi, target_labels] = 1
    tconf = is_obj_mask.float()

    # TODO : return with nicer format
    return b, best_n, gj, gi, target_boxes, target_labels, is_obj_mask, no_obj_mask, tx, ty, tw, th, tc, tconf


def build_targets(pred_shape, pred_cls, targets, anchors, ignore_thresh, device: Union[str, torch.device] = 'cuda'):
    """
    encode label relative image format to yolo grid format
    adapted from :
        https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/47b7c912877ca69db35b8af3a38d6522681b3bb3/utils/utils.py#L267
    """
    if not len(pred_shape) == 3:
        raise RuntimeError(
            "expect pred_shape to have len of 3! got %s" % (len(pred_shape)))
    nB, nA, nG = pred_shape
    class_mask = torch.zeros(nB, nA, nG, nG, device=device)

    encoded_labels = encode_yolo_bbox_labels(
        pred_shape, pred_cls.shape[-1], targets, anchors, ignore_thresh, device)
    b, best_n, gj, gi, target_boxes, target_labels, is_obj_mask, no_obj_mask, tx, ty, tw, th, tc, tconf = encoded_labels
    class_mask[b, best_n, gj, gi] = (
        pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()

    # TODO : return with nicer format
    return class_mask, is_obj_mask, no_obj_mask, tx, ty, tw, th, tc, tconf
