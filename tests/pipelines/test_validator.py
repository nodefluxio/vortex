import numpy as np
from easydict import EasyDict

import pytest

try:
    from vortex.development.core.factory import create_model
    import vortex.development.core.engine as engine
except ImportError:
    # affected by API changes, TODO: fix
    pass

from ..dummy_dataset.utils import dataset


class DummyDataset(dataset.DummyDataset):
    def __init__(self, *args, **kwargs):
        super(type(self),self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        img, label = super(type(self),self).__getitem__(index)
        return img, np.asarray(label)

@pytest.mark.skip(reason="affected by API changes (create_model), no way of currently testing this")
def test_create_validator():
    softmax = dict(
        network_args=dict(
            backbone='shufflenetv2_x1.0',
            n_classes=10,
            freeze_backbone=False,
        ),
        preprocess_args=dict(
            input_size=32,
            input_normalization=dict(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ),
        loss_args=dict(
            reduction='mean'
        ),
        postprocess_args={}
    )
    model = create_model(
        EasyDict(name='softmax', **softmax)
    )
    validator = engine.create_validator(
        model, validation_args={},
        dataset=DummyDataset(), 
    )
    assert isinstance(validator, engine.validator.get_validator('classification'))
    mean_std=dict(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )
    RetinaFace = dict(
        preprocess_args=dict(
            input_size=640,
            input_normalization=mean_std,
        ),
        network_args=dict(
            n_classes=1,
            backbone='shufflenetv2_x1.0',
            pyramid_channels=64,
            aspect_ratios=[1, 2., 3.],
        ),
        loss_args=dict(
            overlap_thresh=0.35,
            cls=2.0, box=1.0,
            ldm=1.0, neg_pos=7,
        ),
        postprocess_args=dict(
            nms=True,
        ),
    )
    model = create_model(
        EasyDict(name='RetinaFace', **RetinaFace)
    )
    validation_args = dict(
        score_threshold=0.2,
        iou_threshold=0.2,
    )
    validator = engine.create_validator(
        model, dataset=DummyDataset(), 
        validation_args=validation_args,
    )
    assert isinstance(validator, engine.validator.get_validator('detection'))