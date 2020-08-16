suported_ops = {}

_REQUIRED_ATTRIBUTES = [
    'get_ops',
    'supported_ops'
]

def register_module(module : str) :
    global suported_ops
    exec('from . import %s' %module)
    module_attributes = eval('%s' %module).__dict__.keys()
    for attribute in _REQUIRED_ATTRIBUTES :
        if not attribute in module_attributes :
            raise RuntimeError("dear maintainer, your module(s) is supposed to have the following attribute(s) : %s; got %s, please check!" %(_REQUIRED_ATTRIBUTES, module_attributes))
    suported_ops[module] = eval('%s' %module)

def get_ops(ops : str, *args, **kwargs) :
    if not ops in suported_ops.keys() :
        raise ValueError("%s ops not supported, avaliable : %s" %(ops, suported_ops.keys()))
    return suported_ops[ops].get_ops(*args, **kwargs)

## for maintainer, register your module here :
register_module('nms')
register_module('multiclass_nms')