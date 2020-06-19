supported_collater = {}
ALL_COLLATERS = []

_REQUIRED_ATTRIBUTES = [
    'create_collater',
    'supported_collater'
]

def register_module(module : str) :
    global supported_collater, ALL_COLLATERS
    ## TODO : consider to check module existence before importing
    exec('from . import %s' %module)
    module = eval('%s' %module)
    module_attributes = module.__dict__.keys()
    for attribute in _REQUIRED_ATTRIBUTES :
        if not attribute in module_attributes :
            raise RuntimeError("dear maintainer, your module(s) is supposed to have the following attribute(s) : %s; got %s, please check!" %(_REQUIRED_ATTRIBUTES, module_attributes))
    supported_collater[module] = module.supported_collater
    ALL_COLLATERS.extend(module.supported_collater)

def create_collater(collater : str, *args, **kwargs) :
    if collater is None:
        return None
    if not collater in ALL_COLLATERS :
        raise KeyError("collater %s not supported, available : %s" %(collater, ALL_COLLATERS))
    for module, collaters in supported_collater.items() :
        if collater in collaters :
            return module.create_collater(collater, *args, **kwargs)
    raise RuntimeError("unexpected error! please report this as bug")

## for maintainer, register your module here :
register_module('darknet')
register_module('ssd')
register_module('retinaface')
