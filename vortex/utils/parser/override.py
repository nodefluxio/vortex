from easydict import EasyDict
from typing import List
from copy import copy


def _get_field(config: EasyDict, field : List[str]) :
    if field :
        return _get_field(getattr(config, field[0]), field[1:])
    else :
        return config

def _has_attr(config: EasyDict, field : str) :
    try :
        c = _get_field(config, field.split('.'))
        return True
    except AttributeError :
        return False

def override_param(config: EasyDict, override: EasyDict):
    override_config = copy(config)
    for key, value in override.items():
        assert _has_attr(override_config, key)
        exec('override_config.{} = value'.format(key))
    return override_config