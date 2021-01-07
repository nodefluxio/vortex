from easydict import EasyDict

from .model import ModelBase


supported_models = {}
all_models = []

_REQUIRED_ATTRIBUTES = [
    'supported_models',
    'create_model_components'
]


def _register_task(task: str):
    global supported_models, all_models
    exec('from . import %s' % task)

    module = eval('%s' % task)
    module_attributes = module.__dict__.keys()
    for attribute in _REQUIRED_ATTRIBUTES:
        if not attribute in module_attributes:
            raise RuntimeError("dear maintainer, your module(s) is supposed to have the following "\
                "attribute(s): %s; got %s, please check!" % (_REQUIRED_ATTRIBUTES, module_attributes))

    supported_models[module] = module.all_models
    all_models.extend(module.all_models)


def create_model_components(model_name: str, preprocess_args: EasyDict, network_args: EasyDict, 
        loss_args: EasyDict, postprocess_args: EasyDict, stage: str='train') -> EasyDict:

    if not model_name in all_models:
        raise KeyError("model '%s' not supported, available: %s" %(model_name, all_models))

    for module, m in supported_models.items():
        if model_name in m:
            return module.create_model_components(model_name, preprocess_args, network_args, loss_args, postprocess_args, stage)
    raise RuntimeError("unexpected error! please report this as bug")


_register_task('detection')
_register_task('classification')

import inspect
from functools import partial

def register_model_(model_name, m):
    """
    Register model with model_name.
    Args:
        model_name: str
        m: module or function, should have `create_model_components`
            function, or is named `create_model_components`
    """
    assert isinstance(model_name, str)
    is_function = inspect.isfunction(m)
    is_module = inspect.ismodule(m)
    assert is_function or is_module
    if is_function:
        module = inspect.getmodule(m)
        assert m.__name__ == 'create_model_components'
    else:
        assert 'create_model_components' in dir(m)
        module = m
    ## TODO: check fn args
    supported_models[module] = model_name
    all_models.append(model_name)
    return m

def register_model(model_name):
    """
    Decorator factory for register model,
    binds model_name to returnded function.
    """
    return partial(register_model_, model_name)

def remove_model(model_name):
    global all_models, supported_models
    if not model_name in all_models:
        return False
    all_models.remove(model_name)
    supported_models = {
        module: model_list for module, model_list in supported_models.items() \
            if model_name not in model_list
    }
    return True