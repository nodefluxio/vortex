from typing import Union
import warnings
import inspect

preproc_registry = {}

_REQUIRED_ATTRIBUTES = [
    'get_preprocess'
]

def register_module_(supported_preprocess: Union[list,str], module) :
    global preproc_registry
    if isinstance(module, str):
        warnings.warn("register module with str is deprecated", warnings.DeprecationWarning)
        exec('from . import f{module}')
        module_attributes = eval('f{module}').__dict__.keys()
        for attribute in _REQUIRED_ATTRIBUTES :
            if not attribute in module_attributes :
                raise RuntimeError("dear maintainer, your module(s) is supposed to have the following attribute(s) : %s; got %s, please check!" %(_REQUIRED_ATTRIBUTES, module_attributes))
        preproc_registry[module] = eval('f{module}')
    else:
        if isinstance(supported_preprocess, str):
            supported_preprocess = [supported_preprocess]
        elif isinstance(supported_preprocess, list):
            assert isinstance(supported_preprocess[0], str), \
                "expect supported_preprocess with \
                    element type of str got {}".format(type(supported_preprocess[0]))
        else:
            msg = "expect supported_preprocess with type of \
                str or list of str got {}".format(type(supported_preprocess))
            raise RuntimeError(msg)
        is_module = inspect.ismodule(module)
        is_function = inspect.isfunction(module)
        assert is_module or is_function, \
            "expect module with type of module or function, \
                got type {}".format(type(module))
        if is_module:
            assert all(attr in dir(module) for attr in _REQUIRED_ATTRIBUTES)
        else:
            function_name = _REQUIRED_ATTRIBUTES[0]
            assert module.__name__ == function_name
        for preproc in supported_preprocess:
            preproc_registry[preproc] = module if inspect.ismodule(module) \
                else inspect.getmodule(module)
    return module

def get_preprocess(preprocess : str, *args, **kwargs) :
    if not preprocess in preproc_registry:
        raise ValueError("%s preprocess not supported, available : %s" %(preprocess, list(preproc_registry.keys())))
    return preproc_registry[preprocess].get_preprocess(preprocess, *args, **kwargs)

## for maintainer, register your module here :
from .normalizer import get_preprocess
# register & provide alias (normalizer a.k.a. normalizer)
register_module_(['normalizer', 'flip_normalize', 'flip_normalizer', 'normalize'], get_preprocess)
