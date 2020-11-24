import torch
import torch.nn as nn

from .base_connector import BackbonePoolConnector
# from ..modules.heads.classification import get_head, all_models, supported_models
from ..modules.losses.classification import ClassificationLoss
from ..modules.postprocess.base_postprocess import SoftmaxPostProcess
from ..modules.preprocess import get_preprocess
from easydict import EasyDict

all_models = ['softmax']
supported_models = ['softmax']

class Classification(BackbonePoolConnector):
    def __init__(self, model_name: str, backbone: dict, n_classes: int, *args, **kwargs):
        super(Classification, self).__init__(backbone, feature_type="classifier", 
            n_classes=n_classes, *args, **kwargs)

        # self.head = get_head(model_name, self.backbone.out_channels, n_classes)
        self.task = "classification"
        self.output_format = {
            "class_label": {"indices": [0], "axis": 0},
            "class_confidence": {"indices": [1], "axis": 0}
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        # x = self.head(x)
        return x


def create_model_components(model_name: str, preprocess_args: EasyDict, network_args: EasyDict, 
        loss_args: EasyDict, postprocess_args: EasyDict, stage: str):
    """ build model components for classification task

    This module will return model components for different stages.

    loss function will use `nn.NLLLoss`.
    
    Args:
        config (EasyDict): model config from config file
    
    Raises:
        KeyError: head model is not available
    
    Returns:
        EasyDict: model components
    """
    if not isinstance(model_name, str):
        raise TypeError("expects string got %s" % type(model_name))
    if not isinstance(preprocess_args, dict):
        raise TypeError("expects `preprocess_args` to have type dict, got %s" % type(preprocess_args))
    if not isinstance(network_args, dict):
        raise TypeError("expects `network_args` to have type dict, got %s" % type(network_args))
    if not isinstance(loss_args, dict):
        raise TypeError("expects `loss_args` to have type dict, got %s" % type(loss_args))
    if not isinstance(postprocess_args, dict):
        raise TypeError("expects `postprocess_args` to have type dict, got %s" % type(postprocess_args))
    if not model_name.lower() in all_models:
        raise KeyError("model %s not supported, available : %s" % (model_name, all_models))

    if "input_normalization" in preprocess_args:
        preprocess_args = preprocess_args.input_normalization
        
    components = {
        'preprocess': get_preprocess('normalizer', **preprocess_args),
        'network': Classification(model_name.lower(), **network_args),
        'postprocess': SoftmaxPostProcess(dim=1)
    }
    if stage == 'train':
        if 'additional_args' in loss_args:
            loss_args.update(loss_args.additional_args)
            loss_args.pop('additional_args')
        components['loss'] = ClassificationLoss(**loss_args)
        components['collate_fn'] = None
    elif stage == 'validate':
        pass
    else:
        raise NotImplementedError("stage other than 'train' and 'export' is not yet "\
            "implemented in classification, got stage: %s" % stage)
    
    return EasyDict(components)
