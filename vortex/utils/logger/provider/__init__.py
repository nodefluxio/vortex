from typing import Union
from easydict import EasyDict
from vortex.utils.logger.base_logger import ExperimentLogger

supported_logger = {}

_REQUIRED_ATTRIBUTES = [
    'create_logger'
]

def register_module(module : str) :
    global supported_logger
    exec('from . import %s' %module)
    module_attributes = eval('%s' %module).__dict__.keys()
    for attribute in _REQUIRED_ATTRIBUTES :
        if not attribute in module_attributes :
            raise RuntimeError("dear maintainer, your logger module(s) is supposed to have the following attribute(s) : %s; got %s, please check!" %(_REQUIRED_ATTRIBUTES, module_attributes))
    supported_logger[module] = eval('%s' %module)

def create_logger(logger: Union[EasyDict], config: Union[EasyDict] = None, *args, **kwargs):
    """ create logger from config
    
    Args:
        logger (EasyDict): logger config from config file, e.g: config.logger
        config (EasyDict): the entire config
    
    Raises:
        ValueError: logger module not supported
    """
    if logger == 'None':
        return DummyLogger(
            fields=['__call__',
                    'log_on_hyperparameters',
                    'log_on_step_update',
                    'log_on_epoch_update',
                    'log_on_model_save',
                    'log_on_validation_result'
                    ]
            )
    if isinstance(logger, dict):
        logger = EasyDict(logger)
    
    if "module" not in logger or "args" not in logger:
        raise RuntimeError("logger config is incomplete, expect to have 'module' and 'args' "\
            "attribute, got %s" % logger)

    provider = logger.module
    provider_args = logger.args
    if not provider in supported_logger:
        raise ValueError("%s logger not supported, available : %s" %(provider, supported_logger.keys()))
    logger = supported_logger[provider].create_logger(provider_args, config, **kwargs)
    return logger

class DummyLogger(ExperimentLogger):
    def __init__(self, fields: list = ['__call__']):
        import types
        for field in fields:
            exec('def %s(self,*args, **kwargs) : pass' % field)
            exec('self.%s = types.MethodType(%s, self)' % (field, field))
        super().__init__()

## for maintainer, register your module here :
register_module('comet_ml')