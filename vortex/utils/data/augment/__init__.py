supported_transforms = {}
ALL_TRANSFORMS = []

_REQUIRED_ATTRIBUTES = [
    'create_transform',
    'supported_transforms'
]


def register_module(module: str):
    global supported_transforms, ALL_TRANSFORMS
    # TODO : consider to check module existence before importing
    exec('from . import %s' % module)
    module = eval('%s' % module)
    module_attributes = module.__dict__.keys()
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
register_module('albumentations')
register_module('nvidia_dali')

