import logging
import warnings
import torch

from copy import deepcopy
from pathlib import Path
from easydict import EasyDict
from typing import Union, Callable, Type, Iterable
from collections import OrderedDict

from vortex.development.networks.models import supported_models
from vortex.development.utils.data.collater import create_collater
from vortex.development.utils.logger import create_logger
from vortex.runtime.factory import create_runtime_model
from vortex.development.exporter.base_exporter import BaseExporter

__all__ = ['create_model',
        #    'create_dataset',
           'create_dataloader',
        #    'create_experiment_logger',
        #    'create_exporter'
]

def create_model(model_config: EasyDict, state_dict: Union[str, dict, Path] = None, stage: str = None) -> torch.nn.Module:
    """Function to create model and it's signature components. E.g. loss function, collate function, etc

    Args:
        model_config (EasyDict): Experiment file configuration at `model` section, as EasyDict object
        state_dict (Union[str, dict, Path], optional): [description]. `model` Pytorch state dictionary or commonly known as weight, 
            can be provided as the path to the file, or the returned dictionary object from `torch.load`.
            If this param is provided, it will override checkpoint specified in the experiment file.
            Defaults to None.
        stage (str, optional): If set to 'train', this will enforce that the model must have `loss` and `collate_fn` attributes,
            hence it will make sure model can be used for training stage. If set to 'validate' it will ignore those requirements 
            but cannot be used in training pipeline, and also may still valid for other pipelines.
            Defaults to 'train'.

    Raises:
        TypeError: Raises if the provided `stage` not in 'train' or 'validate'

    Returns:
        EasyDict: The dictionary containing the model's components
        
    Example:
        The dictionary returned will contain several keys :
        
        - `network` : Pytorch model's object which inherit `torch.nn.Module` class.
        - `preprocess` : model's preprocessing module
        - `postprocess` : model's postprocessing module
        - `loss` : if provided, module for model's loss function
        - `collate_fn` : if provided, module to be embedded to dataloader's `collate_fn` function to modify dataset label's format
            into desirable format that can be accepted by `loss` components

        ```python
        from vortex.development.core.factory import create_model
        from easydict import EasyDict

        model_config = EasyDict({
            'name': 'softmax',
            'network_args': {
                'backbone': 'efficientnet_b0',
                'n_classes': 10,
                'pretrained_backbone': True,
            },
            'preprocess_args': {
                'input_size': 32,
                'input_normalization': {
                'mean': [0.4914, 0.4822, 0.4465],
                'std': [0.2023, 0.1994, 0.2010],
                'scaler': 255,
                }
            },
            'loss_args': {
                'reduction': 'mean'
            }
        })

        model_components = create_model(
            model_config = model_config
        )
        print(model_components.keys())
        ```
    """

    if stage is not None:
        warnings.warn("Deprecated argument 'stage' is used, you can remove this safely.", DeprecationWarning)

    logging.info('Creating Pytorch model from experiment file')

    if "model" in model_config:
        model_config = model_config["model"]
    if "name" not in model_config:
        raise RuntimeError("Config is not valid, expected 'config.model.name' argument but not found, "
            "got model config: {}".format(model_config))

    model_name = model_config.name
    try:
        preprocess_args = model_config.preprocess_args
    except:
        preprocess_args = {}
    try:
        network_args = model_config.network_args
    except:
        network_args = {}
    try:
        loss_args = model_config.loss_args
    except:
        loss_args = {}
    try:
        postprocess_args = model_config.postprocess_args
    except:
        postprocess_args = {}

    model = supported_models[model_name](
        preprocess_args=preprocess_args,
        network_args=network_args,
        loss_args=loss_args,
        postprocess_args=postprocess_args,
    )

    if 'init_state_dict' in model_config or state_dict is not None:
        model_path = None
        if isinstance(state_dict, Path):
            state_dict = str(state_dict)

        # Load state_dict from config if specified in experiment file
        if 'init_state_dict' in model_config and state_dict is None:
            logging.info("Loading state_dict from configuration file: {}".format(model_config.init_state_dict))
            model_path = model_config.init_state_dict
        # If specified using function's parameter, override the experiment config init_state_dict
        elif isinstance(state_dict, str):
            logging.info("Loading state_dict: {}".format(state_dict))
            model_path = state_dict
        elif not isinstance(state_dict, dict):
            ## don't know why you would come to this stage
            raise RuntimeError("'config.model.init_state_dict' or 'state_dict' argument must be set.")

        ckpt = None
        if ('init_state_dict' in model_config and state_dict is None) or isinstance(state_dict, str):
            ckpt = torch.load(model_path, map_location=torch.device('cpu'))
        elif isinstance(state_dict, (OrderedDict, dict)):
            ckpt = state_dict

        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        assert isinstance(state_dict, (OrderedDict, dict))
        model.load_state_dict(state_dict, strict=True) 
    return model


def create_dataset(dataset_config : EasyDict,
                   preprocess_config : EasyDict,
                   stage : str,
                   wrapper_format : str = 'default'
                   ):
    from vortex.development.utils.data.dataset.wrapper import BasicDatasetWrapper, DefaultDatasetWrapper

    dataset_config = deepcopy(dataset_config)
    augmentations = []

    if stage == 'train' :
        if 'name' in dataset_config.train:
            dataset = dataset_config.train.name
        elif 'dataset' in dataset_config.train:
            dataset = dataset_config.train.dataset
        else:
            raise RuntimeError("Dataset name in 'dataset_config.train.name' is not set")
        dataset_args = dataset_config.train.args

        if 'augmentations' in dataset_config.train:
            augmentations = dataset_config.train.augmentations
    elif stage == 'validate':
        if 'name' in dataset_config.eval:
            dataset = dataset_config.eval.name
        elif 'dataset' in dataset_config.eval:
            dataset = dataset_config.eval.dataset
        else:
            raise RuntimeError("Dataset name in 'dataset_config.eval.name' is not set "
                "in dataset_config ({}).".format(dataset_config))
        dataset_args = dataset_config.eval.args
    else:
        raise TypeError('Unknown dataset "stage" argument, got {}, expected "train" or "validate"'%stage)

    if wrapper_format=='default':
        dataset_wrapper = DefaultDatasetWrapper
    elif wrapper_format=='basic':
        dataset_wrapper = BasicDatasetWrapper
    else:
        raise RuntimeError('Unknown dataset `wrapper_format`, should be either "default" or "basic", got {} '.format(wrapper_format))

    return dataset_wrapper(dataset=dataset, stage=stage, preprocess_args=preprocess_config,
                          augmentations=augmentations, dataset_args=dataset_args)

def create_dataloader(dataloader_config : EasyDict,
                      dataset_config : EasyDict, 
                      preprocess_config : EasyDict, 
                      stage : str = 'train',
                      collate_fn : Union[Callable,str,None] = None,
                      ) -> Type[Iterable]:
    """Function to create iterable data loader object

    Args:
        dataloader_config (EasyDict): Experiment file configuration at `dataloader` section, as EasyDict object
        dataset_config (EasyDict): Experiment file configuration at `dataset` section, as EasyDict object
        preprocess_config (EasyDict): Experiment file configuration at `model.preprocess_args` section, as EasyDict object
        stage (str, optional): Specify the experiment stage, either 'train' or 'validate'. Defaults to 'train'.
        collate_fn (Union[Callable,str,None], optional): Collate function to reformat batch data serving. Defaults to None.

    Raises:
        TypeError: Raises if provided `collate_fn` type is neither 'str' (registered in Vortex), Callable (custom function), or None
        RuntimeError: Raises if specified 'dataloader' module is not registered

    Returns:
        Type[Iterable]: Iterable dataloader object which served batch of data in every iteration

    Example:
        ```python
        from vortex.development.core.factory import create_dataloader
        from easydict import EasyDict

        dataloader_config = EasyDict({
            'module': 'PytorchDataLoader',
            'args': {
            'num_workers': 1,
            'batch_size': 4,
            'shuffle': True,
            },
        })

        dataset_config = EasyDict({
            'train': {
                'dataset': 'ImageFolder',
                'args': {
                    'root': 'tests/test_dataset/classification/train'
                },
                'augmentations': [{
                    'module': 'albumentations',
                    'args': {
                        'transforms': [
                        {
                            'transform' : 'RandomBrightnessContrast', 
                            'args' : {
                                'p' : 0.5, 'brightness_by_max': False,
                                'brightness_limit': 0.1, 'contrast_limit': 0.1,
                            }
                        },
                        {'transform': 'HueSaturationValue', 'args': {}},
                        {'transform' : 'HorizontalFlip', 'args' : {'p' : 0.5}},
                        ]
                    }
                }]
            },
        })

        preprocess_config = EasyDict({
            'input_size' : 224,
            'input_normalization' : {
                'mean' : [0.5,0.5,0.5],
                'std' : [0.5, 0.5, 0.5],
                'scaler' : 255
            },
        })

        dataloader = create_dataloader(dataloader_config=dataloader_config,
                                        dataset_config=dataset_config,
                                        preprocess_config = preprocess_config,
                                        collate_fn=None)
        for data in dataloader:
            images,labels = data
        ```
    """
    from vortex.development.utils.data.loader import create_loader, wrapper_format

    dataloader_config = deepcopy(dataloader_config)
    dataset_config = deepcopy(dataset_config)
    preprocess_config = deepcopy(preprocess_config)

    if stage not in ('train', 'validate'):
        raise RuntimeError("Unknown stage of {}, expected value: 'train' or 'validate'"
            .format(stage))

    cfg_field = 'eval' if stage == 'validate' else stage
    if "module" in dataloader_config:
        dataloader_module = dataloader_config.module
        dataloader_args = dataloader_config.args
    elif cfg_field in dataloader_config:
        dataloader_module = dataloader_config[cfg_field]['module']
        dataloader_args = dataloader_config[cfg_field]['args']
    else:
        raise RuntimeError("Dataloader module config for stage {} is not found, please "
            "specify them in 'config.dataloader.module' or 'config.dataloader.{}.module'. "
            "dataloader config: {}".format(stage, cfg_field, dataloader_config))

    # for backward compatibility
    if dataloader_module == 'DataLoader':
        dataloader_module = 'PytorchDataLoader'

    dataset = create_dataset(
        dataset_config=dataset_config, 
        stage=stage, 
        preprocess_config=preprocess_config,
        wrapper_format=wrapper_format[dataloader_module]
    )

    if isinstance(collate_fn, str):
        collater_args = {}
        try:
            collater_args = dataloader_config.collater.args
        except:
            collater_args = {}
        collater_args['dataformat'] = dataset.data_format
        collate_fn = create_collater(collate_fn, **collater_args)
    elif not (hasattr(collate_fn, '__call__') or collate_fn is None):
        raise TypeError("Unknown type of 'collate_fn', should be in the type of string, "
            "Callable, or None. Got {}".format(type(collate_fn)))

    # Re-initialize dataset (Temp workaround), adding `disable_image_auto_pad` to collate_fn object 
    # to disable auto pad augmentation
    disable_auto_pad = (
        collate_fn is not None and
        hasattr(collate_fn, "disable_image_auto_pad") and collate_fn.disable_image_auto_pad
    )
    if disable_auto_pad:
        dataset.disable_image_auto_pad()

    dataloader = create_loader(dataloader_module, dataset, collate_fn=collate_fn, **dataloader_args)
    return dataloader

def create_experiment_logger(config : EasyDict):
    logger = config.logging
    experiment_logger = create_logger(logger,config)

    return experiment_logger

def create_exporter(config: Union[EasyDict,dict], 
                    experiment_name: str, 
                    image_size: int, 
                    output_directory: Union[Path,str]='.') -> Type[BaseExporter]:
    from vortex.development.exporter.onnx import OnnxExporter
    from vortex.development.exporter.torchscript import TorchScriptExporter
    module_map = {
        'onnx': (OnnxExporter, '.onnx'),
        'torchscript': (TorchScriptExporter, '.pt')
    }

    module = config['module']
    output_directory = Path(output_directory)
    filename = experiment_name
    if 'filename' in config['args'] :
        filename = config['args']['filename']
    if isinstance(filename, Path):
        filename = filename.name
    if not filename.endswith(module_map[module][1]):
        filename += module_map[module][1]
    filename = output_directory.joinpath(filename)
    config['args'].update({'filename' : filename})

    assert module in ['onnx', 'torchscript']
    module = module_map[module][0]
    exporter = module(
        image_size=image_size,
        **config['args'],
    )
    return exporter