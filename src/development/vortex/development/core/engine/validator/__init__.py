import torch
from typing import Union
from pathlib import Path
from easydict import EasyDict
from typing import Type
from functools import singledispatch

from vortex.development.predictor import create_predictor
from .base_validator import BaseRuntime, BaseValidator
from .boundingbox_validator import BoundingBoxValidator
from .classification_validator import ClassificationValidator
from vortex.development.core.factory import create_runtime_model

## NOTE : extend validator for specific task here
## TODO : make sure consistent with task name(s) in networks.models
## TODO : for each task, make sure output_format requirements is unique
task_validator_map = dict()

def register_validator(task: str, validator_type : Type[BaseValidator]):
    assert task not in task_validator_map
    task_validator_map.update({task: validator_type})
    return validator_type

def remove_validator(task: str):
    return task_validator_map.pop(task, None)

register_validator('detection', BoundingBoxValidator)
register_validator('classification', ClassificationValidator)

def infer_runtime_task(rt: BaseRuntime) :
    """
    given BaseRuntime class, deduce the prediction task 
    by matching `result_type` (which contains exported output_format)
    with validator output_format requirements
    """
    task = 'unknown'
    # result_fmt = rt.result_type._fields
    result_fmt = rt.output_fields
    task_requirements = {
        task : validator_type.__output_format__
            for task, validator_type in task_validator_map.items()
    }
    for key, reqs in task_requirements.items() :
        if all(fmt in result_fmt for fmt in reqs) :
            task = key
            break
    return task

def get_validator(task):
    return task_validator_map[task]

def create_validator_instance(task, predictor, dataset, **validation_args):
    """
    create instance of validator
    """
    assert task in task_validator_map, "unknown task of '%s'" % task
    validator = None
    validator_type = task_validator_map[task]
    validator = validator_type(
        predictor=predictor,
        dataset=dataset,
        **validation_args
    )
    return validator

@singledispatch
def create_validator(model_components: Union[str,Path], dataset, validation_args, 
                     predictor_args=None, device='cpu'):
    """
    """
    model_components = Path(model_components)
    if not model_components.exists():
        raise RuntimeError("model path {} is not exist, please make sure to specify "
            "the correct path".format(str(model_components)))
    runtime = create_runtime_model(
        runtime=device,
        model_path=model_components
    )
    task = infer_runtime_task(runtime)
    return create_validator_instance(
        task=task, predictor=runtime,
        dataset=dataset, **validation_args
    )

@create_validator.register(dict)
def _(model_components, dataset, validation_args, predictor_args=None, device='cpu'):
    return create_validator(EasyDict(model_components), 
        dataset=dataset, validation_args=validation_args, 
        predictor_args=predictor_args, device=device,
    )

@create_validator.register(EasyDict)
def _(model_components, dataset, validation_args, predictor_args=None, device='cpu'):
    if isinstance(device, str):
        device = torch.device(device)
    elif not isinstance(device, torch.device):
        raise RuntimeError("Unknown data type for device; expected `torch.device` " \
            "or `str`, got %s" % type(device))
    if predictor_args is None:
        predictor_args = {}
    predictor = create_predictor(model_components, **predictor_args).to(device)

    task = model_components.network.task
    return create_validator_instance(
        task=task, predictor=predictor,
        dataset=dataset, **validation_args
    )
