supported_models = {}
all_models = []

_REQUIRED_ATTRIBUTES = [
    'get_head',
    'supported_models'
]


def _register_module(module: str):
    global supported_models, all_models
    # TODO : consider to check module existence before importing
    exec('from . import %s' % module)
    module = eval('%s' % module)
    module_attributes = module.__dict__.keys()
    for attribute in _REQUIRED_ATTRIBUTES:
        if not attribute in module_attributes:
            raise RuntimeError("dear maintainer, your module(s) is supposed to have the following attribute(s) : %s; got %s, please check!" % (
                _REQUIRED_ATTRIBUTES, module_attributes))
    supported_models[module] = module.supported_models
    all_models.extend(module.supported_models)


def get_head(model: str, *args, **kwargs):
    if not model in all_models:
        raise KeyError("model %s not supported, available : %s" %
                       (model, all_models))
    for module, models in supported_models.items():
        if model in models:
            return module.get_head(*args, **kwargs)
    raise RuntimeError("unexpected error! please report this as bug")


# for maintainer, register your module here :
_register_module('yolov3')
