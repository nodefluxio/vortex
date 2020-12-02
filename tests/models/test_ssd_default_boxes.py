import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2].joinpath('src', 'development')))

import torch

from math import sqrt
from torch import tensor, allclose, Size, eq

from vortex.development.networks.models.detection.retinaface import PriorBox, DefaultBox, compute_ssd_anchors

"""
test case for compute_ssd_anchors
"""
def test_compute_ssd_anchors() :
    n_feature_maps, s_min, s_max = 3, 0.2, 0.9
    aspect_ratios = [1., 2., 3.] ## 1./2., 1./3. will be included automatically
    sk, wk, hk, anchors_wh = compute_ssd_anchors(n_feature_maps, aspect_ratios, s_min=s_min, s_max=s_max)
    sk_ = [0.2, 0.55, 0.9]
    anchors_w = [
        [0.20, sqrt(2.) * 0.20, sqrt(1./2.) * 0.20, sqrt(3.) * 0.20, sqrt(1./3.) * 0.20, sqrt(0.2*0.55)],
        [0.55, sqrt(2.) * 0.55, sqrt(1./2.) * 0.55, sqrt(3.) * 0.55, sqrt(1./3.) * 0.55, sqrt(0.55*0.9)],
        [0.90, sqrt(2.) * 0.90, sqrt(1./2.) * 0.90, sqrt(3.) * 0.90, sqrt(1./3.) * 0.90, sqrt(1.25*0.9)],
    ]
    anchors_h = [
        [0.20, 0.20 / sqrt(2.), 0.20 / sqrt(1./2.), 0.20 / sqrt(3.), 0.20 / sqrt(1./3.), sqrt(0.2*0.55)],
        [0.55, 0.55 / sqrt(2.), 0.55 / sqrt(1./2.), 0.55 / sqrt(3.), 0.55 / sqrt(1./3.), sqrt(0.55*0.9)],
        [0.90, 0.90 / sqrt(2.), 0.90 / sqrt(1./2.), 0.90 / sqrt(3.), 0.90 / sqrt(1./3.), sqrt(1.25*0.9)],
    ]
    assert len(sk) == len(wk) == len(hk) == n_feature_maps
    assert allclose(
        tensor(sk), tensor(sk_)
    )
    aspect_ratios = [*aspect_ratios, 1./2., 1./3.]
    for k, (wi, hi) in enumerate(zip(wk, hk)) :
        assert len(wi) == len(hi) == (len(aspect_ratios) + 1)
        assert allclose(
            tensor(wi), tensor(anchors_w[k])
        )
        assert allclose(
            tensor(hi), tensor(anchors_h[k])
        )

def test_ssd_prior_box() :
    voc = {
        'feature_maps': [38, 19, 10, 5],
        'min_dim': 300,
        'steps': [8, 16, 32, 64],
        'min_sizes': [30, 60, 111, 162],
        'max_sizes': [60, 111, 162, 213],
        'aspect_ratios': [[2,3],[2,3],[2,3],[2,3]],
        'variance': [0.1, 0.2],
        'clip': 1,
    }
    default_box = PriorBox(**voc)
    default_boxes = default_box()
    assert torch.all(
        default_boxes >= 0. 
    ) and torch.all(
        default_boxes <= 1.
    )

"""
test case for default box
given :
    - image_size
    - aspect_ratios : list of float (1/ar added automatically)
    - variance (from caffe ssd)
    - steps (aka strides)
    - feature_maps (optional)
    - anchors (optional)
expects :
    - instance of defalut box, when called, return anchors for each maps
"""
def test_default_box_simple() :
    """
    tests for single aspect ratio
    note : in SSD paper, aspect ratio of sqrt(sk_i*sk_i1) is added in addition to aspect ratio of 1
        in other words, if your aspect ratio includes 1, we add an 'augmented' aspect ratio, so total aspect ratio is len(aspect_ratio)+1
    """
    kwargs = {
        'image_size' : 160,
        'steps' : [8, 16, 32],
        'aspect_ratios' : [1],
        'variance' : [0.1, 0.2],
        'clip' : 1,
    }
    default_box = DefaultBox(**kwargs)
    boxes = default_box()
    assert torch.all(eq(
        default_box.feature_maps,
        tensor([20, 10, 5])
    ))
    assert default_box.anchors.dim() == 3
    """
    3 -> number of feature maps
    2 -> n aspect ratio (+1)
    2 -> width and height value
    """
    assert default_box.anchors.size() == Size([3,2,2])
    assert boxes.dim() == 2
    """
    (20*20+10*10+5*5)*2 -> number of feature maps (grids) * number of aspect ratios
    2+2 -> cxywh
    """
    assert boxes.size() == Size([(20*20+10*10+5*5)*2,2+2])
    kwargs = {
        'image_size' : 160,
        'steps' : [8, 16, 32],
        'aspect_ratios' : [2, 3],
        'variance' : [0.1, 0.2],
        'clip' : 1,
    }
    default_box = DefaultBox(**kwargs)
    boxes = default_box()
    assert torch.all(eq(
        default_box.feature_maps,
        tensor([20, 10, 5])
    ))
    assert default_box.anchors.dim() == 3
    """
    3 -> number of feature maps
    4 -> n aspect ratio * 2 (for 1/2 and 1/3)
    2 -> width and height value
    """
    assert default_box.anchors.size() == Size([3,4,2])
    assert boxes.dim() == 2
    """
    (20*20+10*10+5*5)*4 -> number of feature maps (grids) * number of aspect ratios
    2+2 -> cxywh
    """
    assert boxes.size() == Size([(20*20+10*10+5*5)*4,2+2])

def test_default_box_custom_anchors() :
    kwargs = {
        'image_size' : 160,
        'steps' : [8, 16, 32],
        'aspect_ratios' : [2, 3],             ## since we directly specify anchors, ar will be ignored
        'anchors' : [
            [[0.2, 0.2],[0.2,0.1],[0.1,0.2]], ## for 20x20
            [[0.5, 0.5],[0.5,0.2],[0.2,0.5]], ## for 10x10
            [[0.9, 0.9],[0.9,0.5],[0.5,0.9]], ## for 5x5
        ],
        'variance' : [0.1, 0.2],
        'clip' : 1,
    }
    default_box = DefaultBox(**kwargs)
    boxes = default_box()
    assert torch.all(eq(
        default_box.feature_maps,
        tensor([20, 10, 5])
    ))
    assert default_box.anchors.dim() == 3
    assert default_box.anchors.size() == Size([3,3,2])
    assert boxes.dim() == 2
    """
    2+2 -> cxywh
    """
    assert boxes.size() == Size([(20*20+10*10+5*5)*3,2+2])