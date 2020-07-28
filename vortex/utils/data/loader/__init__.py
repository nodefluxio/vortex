import importlib
import inspect
from typing import Union
from types import ModuleType

supported_loaders = {}
wrapper_format = {}
ALL_LOADERS = []

_REQUIRED_ATTRIBUTES = [
    'create_loader',
    'supported_loaders'
]

def register_module(module: Union[str,ModuleType]):
    """
    register a loader module to vortex registry
    Args:
        module: str or module-type to be registered
    """
    global supported_loaders, ALL_LOADERS
    # TODO : consider to check module existence before importing
    if isinstance(module, str):
        # when string is passed, assume it's relative
        module = importlib.import_module(f'.{module}',__package__)
    elif inspect.ismodule(module):
        module = module
    else:
        raise ValueError("unsupported type of `module`")
    module_attributes = dir(module)
    for attribute in _REQUIRED_ATTRIBUTES:
        if not attribute in module_attributes:
            raise RuntimeError("dear maintainer, your module(s) is supposed to have the following attribute(s) : %s; but %s is missing" % (
                _REQUIRED_ATTRIBUTES, attribute))
    supported_loaders[module] = [loader[0] for loader in module.supported_loaders]
    wrapper_format.update({a:b for a,b in module.supported_loaders})
    ALL_LOADERS.extend([loader[0] for loader in module.supported_loaders])


def create_loader(loader: str, *args, **kwargs):
    if not loader in ALL_LOADERS:
        raise KeyError("dataloader %s not supported, available : %s" %
                       (loader, ALL_LOADERS))
    for module, loaders in supported_loaders.items():
        if loader in loaders:
            return module.create_loader(*args, **kwargs)
    raise RuntimeError("unexpected error! please report this as bug")


# for maintainer, register your module here :
# example:
# ```
# from .my_module import my_loader
# register_module(my_loader)
# ```
from . import pytorch_loader
register_module(module=pytorch_loader)
# register_module('pytorch_loader') # still valid

from . import nvidia_dali_loader
register_module(module=nvidia_dali_loader)


