import importlib
import inspect
from typing import Union
from types import ModuleType

supported_transforms = {}
ALL_TRANSFORMS = []

_REQUIRED_ATTRIBUTES = [
    'create_transform',
    'supported_transforms'
]


def register_module(module: Union[str,ModuleType]):
    """
    register a loader module to vortex registry
    Args:
        module: str or module-type to be registered
    """
    global supported_transforms, ALL_TRANSFORMS
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
    supported_transforms[module] = module.supported_transforms
    ALL_TRANSFORMS.extend(module.supported_transforms)


def create_transform(transform: str, *args, **kwargs):
    if not transform in ALL_TRANSFORMS:
        raise KeyError("transfomr %s not supported, available : %s" %
                       (transform, ALL_TRANSFORMS))
    for module, transforms in supported_transforms.items():
        if transform in transforms:
            return module.create_transform(*args, **kwargs)
    raise RuntimeError("unexpected error! please report this as bug")


# for maintainer, register your module here :
# example:
# ```
# from .my_module import my_augmentation
# register_module(my_augmentation)
# ```
from . import albumentations
register_module(module=albumentations)
# register_module('albumentations') # still valid

from . import nvidia_dali
register_module(module=nvidia_dali)
