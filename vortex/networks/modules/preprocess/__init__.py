supported_preprocess = {}

_REQUIRED_ATTRIBUTES = [
    'get_preprocess'
]

def register_module(module : str) :
    global supported_preprocess
    exec('from . import %s' %module)
    module_attributes = eval('%s' %module).__dict__.keys()
    for attribute in _REQUIRED_ATTRIBUTES :
        if not attribute in module_attributes :
            raise RuntimeError("dear maintainer, your module(s) is supposed to have the following attribute(s) : %s; got %s, please check!" %(_REQUIRED_ATTRIBUTES, module_attributes))
    supported_preprocess[module] = eval('%s' %module)

def get_preprocess(preprocess : str, *args, **kwargs) :
    if not preprocess in supported_preprocess.keys() :
        raise ValueError("%s preprocess not supported, available : %s" %(preprocess, supported_preprocess.keys()))
    return supported_preprocess[preprocess].get_preprocess(*args, **kwargs)

## for maintainer, register your module here :
register_module('normalizer')