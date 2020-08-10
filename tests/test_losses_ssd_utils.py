import sys
sys.path.append('vortex/development_package')

import vortex.development.networks.modules.losses.utils.ssd as ssd_utils

import torch
from torch import Tensor, tensor, allclose, Size

"""
test case for point_form
given : 
    - boxes with cxywh format
expects : 
    - boxes with xyxy format
"""
def test_point_form_simple() :
    boxes = [0.5, 0.5, 0.1, 0.1]
    boxes = tensor(boxes).unsqueeze(0)
    tf_boxes = ssd_utils.point_form(boxes)
    assert allclose(
        tf_boxes,
        tensor([0.45, 0.45, 0.55, 0.55])
    )

def test_point_form_batched() :
    boxes = [
        [0.5, 0.5, 0.1, 0.1],
        [0.7, 0.5, 0.2, 0.3],
        [0.9, 0.3, 0.1, 0.3],
    ]
    boxes = tensor(boxes)
    tf_boxes = ssd_utils.point_form(boxes)
    assert allclose(
        tf_boxes,
        tensor([
            [0.45, 0.45, 0.55, 0.55],
            [0.6, 0.35, 0.8, 0.65],
            [0.85, 0.15, 0.95, 0.45]
        ])
    )

"""
test case for center_size
given :
    - boxes with xyxy format
expects :
    - boxes with cxywh format
"""
def test_center_size_simple() :
    boxes = [0.45, 0.45, 0.55, 0.55]
    boxes = tensor(boxes).unsqueeze(0)
    tf_boxes = ssd_utils.center_size(boxes)
    assert allclose(
        tf_boxes,
        tensor(
            [0.5, 0.5, 0.1, 0.1]
        )
    )

def test_center_size_batched() :
    boxes = [
        [0.45, 0.45, 0.55, 0.55],
        [0.6, 0.35, 0.8, 0.65],
        [0.85, 0.15, 0.95, 0.45]
    ]
    boxes = tensor(boxes)
    tf_boxes = ssd_utils.center_size(boxes)
    assert allclose(
        tf_boxes,
        tensor([
            [0.5, 0.5, 0.1, 0.1],
            [0.7, 0.5, 0.2, 0.3],
            [0.9, 0.3, 0.1, 0.3],
        ])
    )

"""
test case for intersect
given :
    - box_a with xyxy format
    - box_b with xyxy format
expects :
    - intersection area for each box in box_a with each box in box_b
"""

def test_intersect_simple() :
    box_a = [0.45, 0.45, 0.55, 0.55]
    box_b = [0.45, 0.45, 0.50, 0.50]
    box_a = tensor(box_a).unsqueeze(0)
    box_b = tensor(box_b).unsqueeze(0)
    intersection = ssd_utils.intersect(box_a, box_b)
    assert intersection.dim() == 2
    assert intersection.size() == Size([1,1])
    assert allclose(
        intersection,
        tensor([0.05*0.05])
    )

def test_intersect_batched() :
    box_a = [
        [0.45, 0.45, 0.55, 0.55],
        [0.6, 0.35, 0.8, 0.65],
        [0.85, 0.15, 0.95, 0.45]
    ]
    box_b = [
        [0.45, 0.45, 0.50, 0.50]
    ]
    box_a = tensor(box_a)
    box_b = tensor(box_b)
    intersection = ssd_utils.intersect(box_a, box_b)
    assert intersection.dim() == 2
    assert intersection.size() == Size([3,1])
    assert allclose(
        intersection,
        tensor([
            [0.0025], # box_a[0] iwth box_b
            [0.0], # box_a[1] with box_b
            [0.0], # box_a[2] with box_b
        ])
    )
    ## setting 2
    box_a = [
        [0.45, 0.45, 0.55, 0.55],
        [0.6, 0.35, 0.8, 0.65],
        [0.85, 0.15, 0.95, 0.45]
    ]
    box_b = [
        [0.45, 0.45, 0.50, 0.50],
        [0.65, 0.45, 0.80, 0.60],
        [0.90, 0.20, 0.95, 0.50],
        [0.45, 0.45, 0.55, 0.55],
    ]
    box_a = tensor(box_a)
    box_b = tensor(box_b)
    intersection = ssd_utils.intersect(box_a, box_b)
    assert intersection.dim() == 2

    assert intersection.size() == Size([3,4])
    assert allclose(
        intersection,
        tensor([
            [0.0025, 0.0, 0.0, 0.01], # box_a[0] iwth box_b
            [0.0, 0.15*0.15, 0.0, 0.0], # box_a[1] with box_b
            [0.0, 0.0, 0.05*0.25, 0.0], # box_a[2] with box_b
        ])
    )

"""
test case for jaccard (iou)
given :
    - box_a with xyxy format
    - box_b with xyxy format
expects :
    - iou for each box in box_a with each box in box_b
"""

def test_jaccard_simple() :
    box_a = [0.45, 0.45, 0.55, 0.55]
    box_b = [0.45, 0.45, 0.50, 0.50]
    box_a = tensor(box_a).unsqueeze(0)
    box_b = tensor(box_b).unsqueeze(0)
    ious = ssd_utils.jaccard(box_a, box_b)
    assert ious.dim() == 2
    assert ious.size() == Size([1,1])
    assert allclose(
        ious,
        tensor([0.05*0.05 / (0.1*0.1 + 0.05*0.05 - 0.05*0.05)])
    )

def test_jaccard_batched() :
    box_a = [
        [0.45, 0.45, 0.55, 0.55],
        [0.6, 0.35, 0.8, 0.65],
        [0.85, 0.15, 0.95, 0.45]
    ]
    box_b = [
        [0.45, 0.45, 0.50, 0.50]
    ]
    box_a = tensor(box_a)
    box_b = tensor(box_b)
    ious = ssd_utils.jaccard(box_a, box_b)
    assert ious.dim() == 2
    assert ious.size() == Size([3,1])
    assert allclose(
        ious,
        tensor([
            [0.0025 / (0.1*0.1)], # box_a[0] iwth box_b
            [0.0], # box_a[1] with box_b
            [0.0], # box_a[2] with box_b
        ])
    )
    ## setting 2
    box_a = [
        [0.45, 0.45, 0.55, 0.55],
        [0.6, 0.35, 0.8, 0.65],
        [0.85, 0.15, 0.95, 0.45]
    ]
    box_b = [
        [0.45, 0.45, 0.50, 0.50],
        [0.65, 0.45, 0.80, 0.60],
        [0.90, 0.20, 0.95, 0.50],
        [0.45, 0.45, 0.55, 0.55],
    ]
    box_a = tensor(box_a)
    box_b = tensor(box_b)
    ious = ssd_utils.jaccard(box_a, box_b)
    assert ious.dim() == 2
    assert ious.size() == Size([3,4])
    assert allclose(
        ious,
        tensor([
            [0.25, 0.0, 0.0, 1.0], # box_a[0] iwth box_b
            [0.0, 0.15*0.15 / (0.2*0.3), 0.0, 0.0], # box_a[1] with box_b
            [0.0, 0.0, 0.05*0.25 / (0.1*0.3 + 0.05*0.3 - 0.05*0.25), 0.0], # box_a[2] with box_b
        ])
    )

"""
test case for encode
given :
    - matched -> gt (xyxy)
    - priors
    - variance -> 1-d tensor of size 2, idx 0 for center, idx 1 for wh
"""
def test_encode_simple() :
    variance = [0.1, 0.2]
    matched = [
        [0.45, 0.45, 0.55, 0.55]
    ]
    """
    let's keep things simple for now, perfect match :
    """
    priors = [
        [0.5, 0.5, 0.1, 0.1]
    ]
    variance = tensor(variance)
    matched = tensor(matched)
    priors = tensor(priors)
    encoded = ssd_utils.encode(matched, priors, variance)
    assert encoded.dim() == 2
    assert encoded.size() == Size([1,4])
    assert not any(torch.isnan(encoded).view(-1)) and not any(torch.isinf(encoded).view(-1))
    assert allclose(
        encoded,
        tensor([0, 0, 0, 0]).float(),
        atol=1e-5,
    )

from vortex.development.networks.models.detection.retinaface import DefaultBox

def test_match_targets() :
    box_kwargs = {
        'image_size' : 160,
        'steps' : [8],
        'aspect_ratios' : [1],
        'variance' : [0.1, 0.2],
        'clip' : 1,
    }
    default_box = DefaultBox(**box_kwargs)
    priors = default_box()
    """
    simulate single gt at the center
    """
    targets = [
        [0.45, 0.45, 0.55, 0.55, 1]
    ]
    targets = tensor(targets).unsqueeze(0)
    n_batch = 1
    gt_loc = targets[0][:, :-1]
    gt_cls = targets[0][:, -1]
    """
    find best default box for each targets
    """
    best_prior_overlap, best_prior_idx, best_truth_overlap, best_truth_idx = ssd_utils.match_targets(gt_loc, priors, 0)
    """
    """
    n_anchors = 2 ## when using ar 1, an augmented ar is automatically added! see ssd paper
    feature_maps = [20] ## stride 8 at 160 img_size
    assert best_prior_overlap.dim() == 1
    assert best_prior_idx.dim() == 1
    assert best_prior_overlap.size() == Size([1])
    assert best_prior_idx.size() == Size([1])
    """
    best_truth_overlap holds ious between priors to gt, dimension : 1, shape : 1, grid_size * n_anchors
    """
    assert best_truth_overlap.dim() == 1
    assert best_truth_idx.dim() == 1
    assert best_truth_idx.size() == Size([(feature_maps[0]**2)*n_anchors])
    assert best_truth_overlap.size() == Size([(feature_maps[0]**2)*n_anchors])
    """
    note : priors is organized as follows
    0 -> t, l, ar1..
    1 -> t+1, l, ar2...    
    """

"""
test case for target matcing
given : 
    - threshold 
    - truths : ground truths, xyxy format, IMPORTANT : xyxy format
    - priors : anchors, cxywh format
    - variances 
    - labels
    - loc_t     (mutable)
    - conf_t    (mutable)
    - idx
expects :
    - loc_t and conf_t filled with encoded targets
"""
def test_match_simple() :
    box_kwargs = {
        'image_size' : 160,
        'steps' : [8, 16, 32],
        'aspect_ratios' : [2, 3],
        'variance' : [0.1, 0.2],
        'clip' : 1,
    }
    default_box = DefaultBox(**box_kwargs)
    priors = default_box()
    targets = [
        [0.5, 0.5, 0.6, 0.6, 1]
    ]
    targets = tensor(targets).unsqueeze(0)
    n_batch = 1
    n_priors = priors.size(0)
    encoded_loc = Tensor(n_batch, n_priors, 4)
    encoded_obj = Tensor(n_batch, n_priors)
    threshold = 0.5
    gt_loc = targets[0][:, :-1]
    gt_cls = targets[0][:, -1]
    ssd_utils.match(threshold, gt_loc, priors, default_box.variance, gt_cls, encoded_loc, encoded_obj, 0)
    assert not any(torch.isnan(encoded_loc).view(-1)) and not any(torch.isinf(encoded_loc).view(-1))
    assert not any(torch.isnan(encoded_obj).view(-1)) and not any(torch.isinf(encoded_obj).view(-1))