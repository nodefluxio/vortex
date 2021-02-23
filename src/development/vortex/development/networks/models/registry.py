import inspect


from .model import ModelBase
from vortex.development.utils.registry import Registry

MODELS = Registry("Models",base_class=ModelBase)
# alias
supported_models = MODELS

# provide alias to support internal use
# e.g. from .registry import register model
register_model = MODELS.register
remove_model = MODELS.pop
