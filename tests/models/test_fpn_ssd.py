import sys
sys.path.insert(0,'src/development')

import torch

from easydict import EasyDict
from torch import Tensor, Size, allclose, tensor

from vortex.development.networks.models.detection.fpn_ssd import FPNSSD, create_model_components
from vortex.development.networks.modules.postprocess.fpn_ssd import FPNSSDDecoder
from vortex.development.networks.modules.losses.ssd import MultiBoxLoss

def test_darknet53_fpn_ssd() :
    strides = [8, 16, 32, 64, 128]
    feature_maps = [80, 40, 20, 10, 5]
    grids = [s**2 for s in feature_maps]
    n_predictions = sum([g*2 for g in grids]) # (n_anchors)
    ssd = FPNSSD(image_size=640, n_classes=4, pyramid_channels=256, backbone='darknet53')
    assert ssd.head.n_anchors == 2
    bounding_boxes, classifications = ssd(torch.rand(1,3,640,640))
    assert bounding_boxes.dim() == 3
    assert classifications.dim() == 3
    assert bounding_boxes.size() == Size([1,n_predictions,4])
    assert classifications.size() == Size([1,n_predictions,4+1])
    """
    eval
    """
    ssd.eval()
    predictions = ssd(torch.rand(1,3,640,640))
    assert predictions.dim() == 3
    assert predictions.size() == Size([1,n_predictions,9])

def test_vgg16_fpn_ssd() :
    strides = [8, 16, 32, 64, 128]
    feature_maps = [80, 40, 20, 10, 5]
    grids = [s**2 for s in feature_maps]
    n_predictions = sum([g*2 for g in grids]) # (n_anchors)
    ssd = FPNSSD(image_size=640, n_classes=4, pyramid_channels=256, backbone='vgg16')
    assert ssd.head.n_anchors == 2
    bounding_boxes, classifications = ssd(torch.rand(1,3,640,640))
    assert bounding_boxes.dim() == 3
    assert classifications.dim() == 3
    assert bounding_boxes.size() == Size([1,n_predictions,4])
    assert classifications.size() == Size([1,n_predictions,4+1])
    """
    eval
    """
    ssd.eval()
    predictions = ssd(torch.rand(1,3,640,640))
    assert predictions.dim() == 3
    assert predictions.size() == Size([1,n_predictions,9])

def test_fpn_ssd_decoder() :
    strides = [8, 16, 32]
    feature_maps = [80, 40, 20]
    grids = [s**2 for s in feature_maps]
    n_predictions = sum([g*2 for g in grids]) # (n_anchors)
    ssd = FPNSSD(image_size=640, n_classes=4, pyramid_channels=256, backbone='darknet53').eval()
    decoder = FPNSSDDecoder(
        priors=ssd.default_boxes,
        variance=ssd.anchor_gen.variance,
        n_classes=ssd.head.n_classes
    )
    decoded = decoder(ssd(torch.rand(1,3,640,640)),tensor([0.5]))
    assert len(decoded) == 4
    bounding_boxes, confidences, classes, detections = decoded
    assert classes.dim() == 1
    assert detections.dim() == 3
    assert confidences.dim() == 1
    assert bounding_boxes.dim() == 2
    assert classes.size() == confidences.size()
    assert classes.size(0) == bounding_boxes.size(0) == detections.size(1)
    assert bounding_boxes.size(1) == 4


def test_ssd_loss() :
    strides = [8, 16, 32]
    feature_maps = [80, 40, 20]
    grids = [s**2 for s in feature_maps]
    n_predictions = sum([g*2 for g in grids]) # (n_anchors)
    ssd = FPNSSD(image_size=640, n_classes=4, pyramid_channels=256, backbone='darknet53')
    loss_fn = MultiBoxLoss(ssd.head.n_classes, 0.5, 3, ssd.default_boxes, ssd.anchor_gen.variance)
    predictions = ssd(torch.rand(1,3,640,640))
    ## IMPORTANT target is xyxy
    gt = [
        [0.5, 0.5, 0.6, 0.6, 1]
    ]
    gt = tensor(gt).unsqueeze(0)
    loss = loss_fn(predictions, gt)
    assert not any(torch.isinf(loss).view(-1)) and not any(torch.isnan(loss).view(-1))

def test_darknet53_fpn_ssd_create_components() :
    preprocess_args, model_args = EasyDict(), EasyDict()
    postprocess_args, loss_args = EasyDict(), EasyDict()
    preprocess_args.input_size = 640
    model_args.backbone = 'darknet53'
    model_args.pyramid_channels = 256
    model_args.n_classes = 4
    loss_args.neg_pos = 3
    loss_args.overlap_thresh = 0.5
    modules = create_model_components(
        preprocess_args=preprocess_args,
        network_args=model_args,
        loss_args=loss_args,
        postprocess_args=postprocess_args
    )
    ssd = modules.network
    loss_fn = modules.loss
    predictions = ssd(torch.rand(1,3,640,640))
    assert len(predictions) ==  2
    gt = [
        [0.5, 0.5, 0.6, 0.6, 1]
    ]
    gt = tensor(gt).unsqueeze(0)
    loss = loss_fn(predictions, gt)
    assert not any(torch.isinf(loss).view(-1)) and not any(torch.isnan(loss).view(-1))

from vortex.development.predictor.base_module import BasePredictor

def test_darknet53_fpn_ssd_predictor() :
    preprocess_args, model_args = EasyDict(), EasyDict()
    postprocess_args, loss_args = EasyDict(), EasyDict()
    preprocess_args.input_size = 640
    model_args.backbone = 'darknet53'
    model_args.pyramid_channels = 256
    model_args.n_classes = 4
    loss_args.neg_pos = 3
    loss_args.overlap_thresh = 0.5
    modules = create_model_components(
        preprocess_args=preprocess_args,
        network_args=model_args,
        loss_args=loss_args,
        postprocess_args=postprocess_args
    )
    ssd = modules.network
    postprocess = modules.postprocess
    predictor = BasePredictor(
        model=ssd,
        postprocess=postprocess
    ).eval()
    detections = predictor(
        torch.rand(1,3,640,640),
        score_threshold=tensor([0.01]),
        iou_threshold=tensor([0.01])
    )
    assert detections[0].dim() == 2