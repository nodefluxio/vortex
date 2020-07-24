from .base_backbone import supported_feature_type, Backbone, ClassifierFeature

supported_models = {}
all_models = []

_REQUIRED_ATTRIBUTES = [
    'get_backbone',
    'supported_models'
]

def register_module(module : str) :
    global supported_models, all_models
    ## TODO : consider to check module existence before importing
    exec('from . import %s' %module)
    module = eval('%s' %module)
    module_attributes = module.__dict__.keys()
    for attribute in _REQUIRED_ATTRIBUTES :
        if not attribute in module_attributes :
            raise RuntimeError("dear maintainer, your module(s) is supposed to have the following "\
                "attribute(s): %s; got %s, please check!" % (_REQUIRED_ATTRIBUTES, module_attributes))
    supported_models[module] = module.supported_models
    all_models.extend(module.supported_models)

def get_backbone(model_name: str, pretrained: bool = False, feature_type: str = "tri_stage_fpn", 
                 n_classes: int = 1000, **kwargs):
    if feature_type not in supported_feature_type:
        raise RuntimeError("invalid 'feature_type' value of {}, available [{}]".format(
            feature_type, ", ".join(supported_feature_type)))
    if not model_name in all_models :
        raise KeyError("backbones '%s' is not supported, available: %s" %(model_name, all_models))

    for module, models in supported_models.items():
        if model_name in models:
            return module.get_backbone(model_name, pretrained=pretrained, feature_type=feature_type, 
                 n_classes=n_classes, **kwargs)
    raise RuntimeError("unexpected error! please report this as bug")

## for maintainer, register your module here :
register_module('darknet53')
register_module('efficientnet')
register_module('mobilenetv2')
register_module('mobilenetv3')
register_module('resnet')
register_module('shufflenetv2')
register_module('vgg')

import inspect
from functools import partial

def register_backbone_(model_name, m):
    """
    Register backbone with given model name.
    Args:
        model_name: string or sequence
        m: module or function, should have `create_model_components`
            function, or is named `create_model_components`
    """
    assert isinstance(model_name, (tuple,list,str))
    is_function = inspect.isfunction(m)
    is_module = inspect.ismodule(m)
    assert is_function or is_module
    if is_function:
        module = inspect.getmodule(m)
        assert m.__name__ == 'get_backbone'
    else:
        assert 'get_backbone' in dir(m)
        module = m
    supported_models[module] = [model_name] \
        if isinstance(model_name,str) else model_name
    all_models.extend(supported_models[module])
    return m

def register_backbone(model_name):
    """
    Decorator factory for register backbone, 
    binds model_name to returned function.
    """
    return partial(register_backbone_, model_name)

def remove_backbone(model_name):
    global all_models, supported_models
    if not model_name in all_models:
        return False
    for k, model_list in supported_models.items():
        if model_name in model_list:
            for m in model_list:
                all_models.remove(m)
            break
    supported_models = {
        module: model_list for module, model_list in supported_models.items() \
            if model_name not in model_list
    }
    return True