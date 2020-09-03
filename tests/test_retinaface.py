import sys
sys.path.insert(0,'src/development')

import torch
from easydict import EasyDict
from torch import Tensor, Size, allclose, tensor

from vortex.development.networks.modules.postprocess.retinaface import RetinaFaceDecoder
from vortex.development.networks.models.detection.retinaface import RetinaFace, create_model_components

def test_darknet53_retinaface() :
    """
    training
    """
    retinaface = RetinaFace(image_size=640,backbone='darknet53', pyramid_channels=256)
    bounding_boxes, classifications, landmarks = retinaface(torch.rand(1,3,640,640))
    n_anchors = 2
    assert retinaface.head.n_anchors == n_anchors
    feature_maps = [80, 40, 20]
    grid_size = [f ** 2 for f in feature_maps]
    feature_shape = [g * n_anchors for g in grid_size]
    predcition_channels = sum(feature_shape)
    assert landmarks.dim() == 3
    assert bounding_boxes.dim() == 3
    assert classifications.dim() == 3
    assert landmarks.size() == Size([1,predcition_channels,10])
    assert bounding_boxes.size() == Size([1,predcition_channels,4])
    assert classifications.size() == Size([1,predcition_channels,2])

    """
    eval
    """
    retinaface.eval()
    predictions = retinaface(torch.rand(1,3,640,640))
    assert predictions.dim() == 3
    assert predictions.size() == Size([1,predcition_channels,16])

def test_darknet53_retinaface_decoder() :
    retinaface = RetinaFace(image_size=640, backbone='darknet53', pyramid_channels=256).eval()
    predictions = retinaface(torch.rand(1,3,640,640))
    decoder = RetinaFaceDecoder(retinaface.default_boxes,retinaface.anchor_gen.variance,n_landmarks=retinaface.n_landmarks)
    decoded = decoder(predictions, tensor([0.5]))
    assert len(decoded) == 4
    bounding_boxes, confidences, classes, detections = decoded
    assert classes.dim() == 1
    assert detections.dim() == 3
    assert confidences.dim() == 1
    assert bounding_boxes.dim() == 2
    assert classes.size() == confidences.size()
    assert classes.size(0) == bounding_boxes.size(0) == detections.size(1)
    assert bounding_boxes.size(1) == 4

def test_darknet53_retinaface_create_components() :
    preprocess_args, model_args = EasyDict(), EasyDict()
    postprocess_args, loss_args = EasyDict(), EasyDict()
    preprocess_args.input_size = 640
    model_args.backbone = 'darknet53'
    model_args.pyramid_channels = 256
    loss_args.neg_pos = 3
    loss_args.overlap_thresh = 0.5
    modules = create_model_components(
        preprocess_args=preprocess_args,
        network_args=model_args,
        loss_args=loss_args,
        postprocess_args=postprocess_args
    )
    retinaface = modules.network
    loss_fn = modules.loss
    predictions = retinaface(torch.rand(1,3,640,640))
    assert len(predictions) == 3
    gt = [[
        0.5, 0.5, 0.6, 0.6, 0.4, 0.4, 0.6, 0.6, 0.4, 0.4, 0.6, 0.6, 0.4, 0.4, 0
    ]]
    gt = tensor(gt).unsqueeze(0)
    loss = loss_fn(predictions, gt)
    assert not any(torch.isinf(loss).view(-1)) and not any(torch.isnan(loss).view(-1))
    retinaface.eval()
    assert retinaface(torch.rand(1,3,640,640)).size(2) == 16

from vortex.development.networks import models

def test_create_model_components() :
    preprocess_args, model_args = EasyDict(), EasyDict()
    postprocess_args, loss_args = EasyDict(), EasyDict()
    preprocess_args.input_size = 640
    model_args.backbone = 'darknet53'
    model_args.pyramid_channels = 256
    loss_args.neg_pos = 3
    loss_args.overlap_thresh = 0.5
    model_name, stage = 'RetinaFace', 'train'
    modules = models.create_model_components(
        model_name=model_name,
        preprocess_args=preprocess_args,
        network_args=model_args,
        loss_args=loss_args,
        postprocess_args=postprocess_args,
        stage=stage,
    )
    retinaface = modules.network
    loss_fn = modules.loss
    predictions = retinaface(torch.rand(1,3,640,640))
    assert len(predictions) == 3
    gt = [[
        0.5, 0.5, 0.6, 0.6, 0.4, 0.4, 0.6, 0.6, 0.4, 0.4, 0.6, 0.6, 0.4, 0.4, 0
    ]]
    gt = tensor(gt).unsqueeze(0)
    loss = loss_fn(predictions, gt)
    assert not any(torch.isinf(loss).view(-1)) and not any(torch.isnan(loss).view(-1))
    retinaface.eval()
    assert retinaface(torch.rand(1,3,640,640)).size(2) == 16

from vortex.development.predictor.base_module import BasePredictor

def test_darknet53_retinaface_predictor() :
    preprocess_args, model_args = EasyDict(), EasyDict()
    postprocess_args, loss_args = EasyDict(), EasyDict()
    preprocess_args.input_size = 640
    model_args.backbone = 'darknet53'
    model_args.pyramid_channels = 256
    loss_args.neg_pos = 3
    loss_args.overlap_thresh = 0.5
    model_name, stage = 'RetinaFace', 'train'
    modules = models.create_model_components(
        model_name=model_name,
        preprocess_args=preprocess_args,
        network_args=model_args,
        loss_args=loss_args,
        postprocess_args=postprocess_args,
        stage=stage,
    )
    retinaface = modules.network
    postprocess = modules.postprocess
    predictor = BasePredictor(
        model=retinaface,
        postprocess=postprocess
    ).eval()
    detections = predictor(
        torch.rand(1,3,640,640),
        tensor([0.01]),
        tensor([0.01])
    )
    assert detections[0].dim() == 2