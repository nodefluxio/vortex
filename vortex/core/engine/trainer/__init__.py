from typing import Type, Union
from copy import deepcopy
from easydict import EasyDict

from .default_trainer import DefaultTrainer
from .base_trainer import BaseTrainer

trainer_map = {}

def register_trainer(trainer_type: type, name=None):
    assert isinstance(trainer_type, type)
    assert isinstance(name, str) or name is None
    name = name if name is not None \
        else trainer_type.__name__
    trainer_map.update({name : trainer_type})
    return trainer_type

def remove_trainer(trainer: str):
    return trainer_map.pop(trainer, None)

register_trainer(DefaultTrainer)

def create_trainer(trainer_config: Union[dict, EasyDict], **kwargs):
    trainer_config = EasyDict(trainer_config)
    assert hasattr(trainer_config, 'driver')
    assert hasattr(trainer_config, 'optimizer') \
        or 'optimizer' in kwargs
    if hasattr(trainer_config, 'optimizer'):
        optimizer = deepcopy(trainer_config.optimizer)
    else:
        optimizer = kwargs.pop('optimizer')
    if hasattr(trainer_config, 'scheduler'):
        scheduler = deepcopy(trainer_config.scheduler)
    else:
        scheduler = None
    driver = deepcopy(trainer_config.driver)
    trainer = driver.module
    trainer_args = driver.args
    trainer_args.update(kwargs)
    trainer_args.update(dict(
        optimizer=optimizer,
        scheduler=scheduler,
    ))
    if not trainer in trainer_map:
        raise RuntimeError("unsupported train driver %s, supported : %s" % (
            trainer, ','.join(trainer_map.keys())))
    trainer = trainer_map[trainer]
    return trainer(**trainer_args)