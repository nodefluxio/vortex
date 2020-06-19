import torch
import torch.nn as nn

from torch import tensor

from ...modules.heads.detection.fpn import FPNBackbone
from ...modules.heads.ssh import SSHLandmarkDetectionModule, SSHLandmarkDetectionHead

import warnings

from math import sqrt
from itertools import product
from typing import List, Tuple, Union, Sequence


"""
# SSD300 CONFIGS
voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}
"""

def compute_ssd_anchors(n_feature_maps : int, aspect_ratios : List[float], s_min : float=0.2, s_max : float=0.9) :
    """
    ssd multibox detector equation no 4,
    assuming equally spaced feature maps in between, e.g. with stride 8, 16, 32, ...
    """
    sk, wk, hk = [], [], []
    m = n_feature_maps
    anchors_wh = []
    for i in range(m) :
        k = i + 1
        sk_i  = s_min + (s_max - s_min) * (k - 1) / ((m - 1) if m > 1 else 1)
        sk_i1 = s_min + (s_max - s_min) * (k) / ((m - 1) if m > 1 else 1)
        sk_prime = sqrt(sk_i * sk_i1)
        wk_i, hk_i = [], []
        for ar in aspect_ratios :
            w, h = sk_i * sqrt(ar), sk_i / sqrt(ar)
            wk_i.append(w)
            hk_i.append(h)
            if ar - 1 > 0 :
                wk_i.append(h)
                hk_i.append(w)
        if 1 in aspect_ratios :
            wk_i.append(sk_prime)
            hk_i.append(sk_prime)
        anchors_wh.append(list(zip(wk_i, hk_i)))
        sk.append(sk_i)
        wk.append(wk_i)
        hk.append(hk_i)
    return sk, wk, hk, anchors_wh


class PriorBox(nn.Module):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    adapted from :
        https://github.com/amdegroot/ssd.pytorch/blob/8dd38657a3b1df98df26cf18be9671647905c2a0/layers/functions/prior_box.py
    """
    __constants__ = ['min_dim', 'aspect_ratios', 'variance', 'feature_maps', 'min_sizes', 'max_sizes', 'steps', 'clip']
    def __init__(self, min_dim : int, aspect_ratios : List[List[int]], variance : List[int], feature_maps : List[int], min_sizes : List[int], max_sizes : List[int], steps : List[int], clip : int):
        super(PriorBox, self).__init__()
        self.register_buffer('min_dim', tensor([min_dim]).float())
        self.register_buffer('aspect_ratios', tensor(aspect_ratios).float())
        self.register_buffer('variance', tensor(variance).float())
        self.register_buffer('feature_maps', tensor(feature_maps).long())
        self.register_buffer('min_sizes', tensor(min_sizes).float())
        self.register_buffer('max_sizes', tensor(max_sizes).float())
        self.register_buffer('steps', tensor(steps).long())
        self.register_buffer('clip', tensor([clip]).long())
        self.image_size = self.min_dim

        ## self.image_size = cfg['min_dim']
        ## # number of priors for feature map location (either 4 or 6)
        ## self.num_priors = len(cfg['aspect_ratios'])
        ## self.variance = cfg['variance'] or [0.1]
        ## self.feature_maps = cfg['feature_maps']
        ## self.min_sizes = cfg['min_sizes']
        ## self.max_sizes = cfg['max_sizes']
        ## self.steps = cfg['steps']
        ## self.aspect_ratios = cfg['aspect_ratios']
        ## self.clip = cfg['clip']
        ## self.version = cfg['name']
        ## for v in self.variance:
        ##     if v <= 0:
        ##         raise ValueError('Variances must be greater than 0')

    def forward(self) -> torch.Tensor:
        ## create tile
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip.item() :
            ## TODO : clip xyxy
            output.clamp_(max=1, min=0)
        return output

class DefaultBox(nn.Module):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Computes default box from SSD or use user-specified anchors
    adapted from :
        https://github.com/amdegroot/ssd.pytorch/blob/8dd38657a3b1df98df26cf18be9671647905c2a0/layers/functions/prior_box.py
    """
    __constants__ = ['image_size', 'aspect_ratios', 'variance', 'feature_maps', 'min_sizes', 'max_sizes', 'steps', 'clip', 'anchors']
    def __init__(self, image_size : int, aspect_ratios : List[float], variance : List[int], steps : List[int], feature_maps : Union[List[int]]=None, anchors : Union[Sequence[Sequence[Sequence[int]]]]=None, clip : int=1, s_min=None, s_max=None):
        super(DefaultBox, self).__init__()
        if anchors is None :
            assert all([ar >= 1 for ar in aspect_ratios]), "when using `aspect_ratio` to compute anchors, please specify >= 1 aspect ratio only, i.e. 1, 2, 3; 1./2, 1./3 will be added automatically"
            kwargs = {}
            if s_min :
                kwargs['s_min'] = s_min
            if s_max :
                kwargs['s_max'] = s_max
            sk, wk, hk, anchors_wh = compute_ssd_anchors(n_feature_maps=len(steps),aspect_ratios=aspect_ratios,**kwargs)
            anchors = anchors_wh
        else :
            ignored = ['aspect_ratios']
            ignored = ignored + [str(s_min)] if s_min else ignored
            ignored = ignored + [str(s_max)] if s_max else ignored
            warnings.warn('ignoring `%s` for computing anchors, directly use `acnhors` instead' %', '.join(ignored))
            warnings.warn('when using custom anchors, we assume your anchors is relative')
        if feature_maps is None :
            warnings.warn('inferring feature maps size from image size and stride')
            feature_maps = [image_size // stride for stride in steps]
        self.register_buffer('image_size', tensor([image_size]).float())
        self.register_buffer('aspect_ratios', tensor(aspect_ratios).float())
        self.register_buffer('variance', tensor(variance).float())
        self.register_buffer('feature_maps', tensor(feature_maps).long())
        self.register_buffer('anchors', tensor(anchors).float())
        self.register_buffer('steps', tensor(steps).long())
        self.register_buffer('clip', tensor([clip]).long())

        assert len(self.anchors) == len(feature_maps)
        assert self.anchors.dim() == 3, "when using custom anchors, we expects 3-d array-like"
        assert self.anchors.size(2) == 2

    def forward(self) -> torch.Tensor:
        ## create tile
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                for anchor in self.anchors[k] :
                    w, h = anchor
                    mean.append([cx, cy, w, h])
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip.item() :
            output.clamp_(max=1, min=0)
        return output

class RetinaFace(FPNBackbone) :
    __constants__ = [
        'default_boxes',
    ]
    def __init__(self, image_size : int, n_landmarks : int=5, aspect_ratios : List[float]=[1.], variance : List[int]=[0.1,0.2], anchors : Union[Sequence[Sequence[Sequence[int]]]]=None, *args, **kwargs) :
        super(RetinaFace, self).__init__(*args, **kwargs)
        fpn_channels = self.fpn.output_channels

        strides = [8,16,32]
        anchor_gen =  DefaultBox(
            image_size=image_size,
            aspect_ratios=aspect_ratios,
            variance=variance,
            steps=strides,
            anchors=anchors
        )
        self.anchor_gen = anchor_gen
        self.register_buffer('default_boxes', self.anchor_gen())

        n_anchors : int = len(self.anchor_gen.anchors[0])
        self.n_landmarks = n_landmarks

        head_module = SSHLandmarkDetectionHead(fpn_channels, n_anchors=n_anchors, n_landmarks=n_landmarks)
        self.add_module('head', head_module)

        self.output_format = dict(
            bounding_box=dict(
                indices=[0,1,2,3],
                axis=1,
            ),
            class_label=dict(
                indices=[5],
                axis=1,
            ),
            class_confidence=dict(
                indices=[4],
                axis=1,
            ),
            landmarks=dict(
                indices=[i+6 for i in range(n_landmarks*2)],
                axis=1,
            )
        )
        self.task = "detection"


    def forward(self, x) :
        p3, p4, p5 = super(RetinaFace, self).forward(x)
        predictions = self.head(p3, p4, p5)
        return predictions

supported_models = [
    'RetinaFace'
]

from easydict import EasyDict
from ...modules.losses.ssd import MultiBoxLandmarkLoss
from ...modules.postprocess.retinaface import RetinaFacePostProcess

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight.data)
        if m.bias is not None :
            m.bias.data.zero_()

def create_model_components(preprocess_args: EasyDict, network_args: EasyDict, loss_args: EasyDict, postprocess_args: EasyDict) -> EasyDict:
    assert hasattr(preprocess_args, 'input_size')
    model_components = EasyDict()
    network_args.image_size = preprocess_args.input_size
    network = RetinaFace(**network_args)
    network.head.apply(weights_init)
    loss_args.num_classes = 2
    loss_args.priors = network.default_boxes
    loss_args.n_landmarks = network.n_landmarks
    loss_fn = MultiBoxLandmarkLoss(**loss_args)
    postprocess_args.variance = network.anchor_gen.variance
    postprocess_args.priors = network.default_boxes
    postprocess_args.n_landmarks = network.n_landmarks
    postprocess = RetinaFacePostProcess(**postprocess_args)
    model_components.network = network
    model_components.loss = loss_fn
    model_components.postprocess = postprocess
    model_components.collate_fn = 'RetinaFaceCollate'
    return model_components
