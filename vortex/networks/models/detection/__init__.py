from typing import Union, List, Dict, Tuple
from easydict import EasyDict

supported_models = {}
all_models = []

_REQUIRED_ATTRIBUTES = [
    'supported_models',
    'create_model_components'
]


def register_module(module: str):
    global supported_models, all_models
    exec('from . import %s' % module)
    module = eval('%s' % module)
    module_attributes = module.__dict__.keys()
    for attribute in _REQUIRED_ATTRIBUTES:
        if not attribute in module_attributes:
            raise RuntimeError("dear maintainer, your module(s) is supposed to have the following\
                attribute(s): %s; got %s, please check!" % (_REQUIRED_ATTRIBUTES, module_attributes))
    supported_models[module] = module.supported_models
    all_models.extend(module.supported_models)


def create_model_components(model_name: str, preprocess_args: EasyDict, network_args: EasyDict, 
        loss_args: EasyDict, postprocess_args: EasyDict, stage: str):
    """ Building model components for different stages of usage.
    
    There are 2 stages that is supported: `train` and `export`. 
    The model components is returned as an `EasyDict` module.

    The `train` stages contains: `network`, `loss`, `collate_fn`, `postprocess` module.
    The `export` stages contains: `network`, `postprocess` module.
    
    Args:
        model_name (str): model name to be built. see available models (###input context here).
        preprocess_args (EasyDict): pre-process options for input image from config, see (###input context here).
        network_args (EasyDict): arguments to build backbone from config, see (###input context here).
        loss_args (EasyDict): arguments to build loss from config, see (###input context here).
        postprocess_args (EasyDict): arguments to build post-process stage from config, see (###input context here).
        stage (str): what stage to build the models, available: `train` and `export`.
    
    Raises:
        KeyError: raised when `model_name` is not available.
    
    Returns:
        EasyDict: the build model components.
    """
    if not isinstance(model_name, str):
        raise TypeError("expects string got %s" % type(model_name))
    if not isinstance(preprocess_args, dict):
        raise TypeError("expects string got %s" % type(preprocess_args))
    if not isinstance(network_args, dict):
        raise TypeError("expects string got %s" % type(network_args))
    if not isinstance(loss_args, dict):
        raise TypeError("expects string got %s" % type(loss_args))
    if not isinstance(postprocess_args, dict):
        raise TypeError("expects string got %s" % type(postprocess_args))
    if not model_name in all_models:
        raise KeyError("model %s not supported, available : %s" %
                       (model_name, all_models))
    for module, models in supported_models.items():
        if model_name in models:
            from ...modules.preprocess import get_preprocess
            model_components = module.create_model_components(
                preprocess_args, network_args, loss_args, postprocess_args)
            if not isinstance(model_components, EasyDict):
                raise TypeError(
                    "return type from function create_model_components expected as EasyDict, got %s" % type(model_components))
            if stage == 'train':
                key_components = ['network', 'loss', 'collate_fn', 'postprocess']
                check_model_components_keys(
                    stage=stage, model_name=model_name,
                    key_components=key_components,
                    returned_components=model_components
                )
                if 'optimizer_params' in model_components:
                    key_components.append('optimizer_params')
            elif stage == 'validate':
                key_components = ['network', 'postprocess']
                check_model_components_keys(
                    stage=stage, model_name=model_name,
                    key_components=key_components,
                    returned_components=model_components)
            model_components = EasyDict({component: model_components[component] for component in key_components})

            preprocess_args = {}
            if "input_normalization" in preprocess_args:
                preprocess_args = preprocess_args.input_normalization
            model_components.preprocess = get_preprocess('normalizer', **preprocess_args)
            model_components.network.task = 'detection' ## force
            if not hasattr(model_components.network, "output_format"):
                raise RuntimeError("model {} doesn't have 'output_format' attribute".format(model_name))
            return model_components
    raise RuntimeError("unexpected error! please report this as bug")


def check_model_components_keys(stage: str, model_name: str, 
        key_components: List, returned_components: EasyDict):

    missing_keys = []
    for key in key_components:
        if key not in returned_components.keys():
            missing_keys.append(key)
    if len(missing_keys) > 0:
        raise RuntimeError(
            "dear maintainer, your returned value of create_model_components for stage '%s' on model '%s' is missing the following key(s) : %s" % (stage, model_name, repr(missing_keys)))


# for maintainer, register your module here :
register_module('yolov3')
register_module('fpn_ssd')
register_module('retinaface')
register_module('detr')
