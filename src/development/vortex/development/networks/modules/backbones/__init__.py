import warnings
from vortex.development.utils.registry import Registry

from .base_backbone import BackboneBase, BackboneConfig
from . import (
    cspnet, darknet, efficientnet, mobilenetv2, mobilenetv3,
    regnet, resnest, resnet, rexnet, shufflenetv2, tresnet, vgg
)

BACKBONES = Registry("Backbones")

supported_models = {}
all_models = []

_REQUIRED_ATTRIBUTES = [
    'supported_models'
]

register_backbone = BACKBONES.register
remove_backbone = BACKBONES.pop


def register_module(module):
    global supported_models, all_models
    module_attributes = module.__dict__.keys()
    for attribute in _REQUIRED_ATTRIBUTES:
        if not attribute in module_attributes :
            raise RuntimeError("dear maintainer, your module(s) is supposed to have the following "
                "attribute(s): %s; got %s, please check!" % (_REQUIRED_ATTRIBUTES, module_attributes))
    for bb in module.supported_models:
        fn = getattr(module, bb)
        register_backbone(module=fn, name=bb)
    supported_models[module] = module.supported_models
    all_models.extend(module.supported_models)


def get_backbone(model_name: str, pretrained: bool = False, feature_type=None, n_classes: int = 1000, **kwargs):
    if feature_type is not None:
        warnings.warn("'feature_type' in 'get_backbone' is now deprecated and not used here, please use "
            "'vortex.development.networks.Backbone' to specify stages output", DeprecationWarning)
    if not model_name in BACKBONES:
        raise KeyError("backbones '{}' is not supported, available: {}".format(model_name, all_models))

    kwargs.update(num_classes=n_classes, pretrained=pretrained)
    return BACKBONES.create_from_args(model_name, **kwargs)


## for maintainer, register your module here :
register_module(darknet)
register_module(cspnet)
register_module(efficientnet)
register_module(mobilenetv2)
register_module(mobilenetv3)
register_module(regnet)
register_module(resnet)
register_module(resnest)
register_module(rexnet)
register_module(shufflenetv2)
register_module(tresnet)
register_module(vgg)
