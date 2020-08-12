# SUPPORTED_METHODS = {}
# ALL_LOSS = []

# _REQUIRED_ATTRIBUTES = [
#     'create_loss',
#     'SUPPORTED_METHODS'
# ]


# def register_module(module: str):
#     global SUPPORTED_METHODS, ALL_LOSS
#     # TODO : consider to check module existence before importing
#     exec('from . import %s' % module)
#     module = eval('%s' % module)
#     module_attributes = module.__dict__.keys()
#     for attribute in _REQUIRED_ATTRIBUTES:
#         if not attribute in module_attributes:
#             raise RuntimeError("dear maintainer, your module(s) is supposed to have the following attribute(s) : %s; got %s, please check!" % (
#                 _REQUIRED_ATTRIBUTES, module_attributes))
#     SUPPORTED_METHODS[module] = module.SUPPORTED_METHODS
#     ALL_LOSS.extend(module.SUPPORTED_METHODS)


# def create_loss(method: str, *args, **kwargs):
#     if not method in ALL_LOSS:
#         raise KeyError("method %s not supported, available : %s" %
#                        (method, SUPPORTED_METHODS))
#     for module, methods in SUPPORTED_METHODS.items():
#         if method in methods:
#             return module.create_loss(*args, **kwargs)
#     raise RuntimeError("unexpected error! please report this as bug")


# # for maintainer, register your module here :
# register_module('yolov3')
