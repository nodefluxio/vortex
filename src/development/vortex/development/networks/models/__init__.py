## this is required to register model
from . import classification
from . import detection

from .backbone import Backbone, ModuleIOHook
from .model import ModelBase
from .registry import MODELS, supported_models, register_model, remove_model

