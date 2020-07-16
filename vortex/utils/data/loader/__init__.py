supported_loaders = {}
ALL_LOADERS = []

_REQUIRED_ATTRIBUTES = [
    'create_loader',
    'supported_loaders'
]

def register_module(module: str):
    global supported_loaders, ALL_LOADERS
    # TODO : consider to check module existence before importing
    exec('from . import %s' % module)
    module = eval('%s' % module)
    module_attributes = module.__dict__.keys()
    for attribute in _REQUIRED_ATTRIBUTES:
        if not attribute in module_attributes:
            raise RuntimeError("dear maintainer, your module(s) is supposed to have the following attribute(s) : %s; but %s is missing" % (
                _REQUIRED_ATTRIBUTES, attribute))
    supported_loaders[module] = module.supported_loaders
    ALL_LOADERS.extend(module.supported_loaders)


def create_loader(loader: str, *args, **kwargs):
    if not loader in ALL_LOADERS:
        raise KeyError("dataloader %s not supported, available : %s" %
                       (loader, ALL_LOADERS))
    for module, loaders in supported_loaders.items():
        if loader in loaders:
            return module.create_loader(*args, **kwargs)
    raise RuntimeError("unexpected error! please report this as bug")


# for maintainer, register your module here :
register_module('pytorch_loader')
