SUPPORTED_POSTPROCESS = {}

_REQUIRED_ATTRIBUTES = [
    'get_postprocess'
]


def register_module(module: str):
    global SUPPORTED_POSTPROCESS
    exec('from . import %s' % module)
    module_attributes = eval('%s' % module).__dict__.keys()
    for attribute in _REQUIRED_ATTRIBUTES:
        if not attribute in module_attributes:
            raise RuntimeError("dear maintainer, your module(s) is supposed to have the following attribute(s) : %s; got %s, please check!" % (
                ', '.join(_REQUIRED_ATTRIBUTES), ', '.join(module_attributes)))
    SUPPORTED_POSTPROCESS[module] = eval('%s' % module)


def get_postprocess(postprocess: str, *args, **kwargs):
    if not postprocess in SUPPORTED_POSTPROCESS.keys():
        raise ValueError("%s postprocess not supported, avaliable : %s" % (
            postprocess, SUPPORTED_POSTPROCESS.keys()))
    return SUPPORTED_POSTPROCESS[postprocess].get_postprocess(*args, **kwargs)


# for maintainer, register your module here :
register_module('yolov3')
