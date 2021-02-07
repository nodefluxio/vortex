import inspect
from functools import partial


from .model import ModelBase
from vortex.development.utils.registry import Registry

# TODO: naming fix
model_registry = Registry("model",base_class=ModelBase)
supported_models = model_registry

# provide alias to support internal use
# e.g. from .registry import register model
register_model = model_registry.register
remove_model = model_registry.pop
