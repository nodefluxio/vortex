import os
import sys
import logging
import warnings
import torch

from copy import deepcopy
from pathlib import Path
from easydict import EasyDict
from typing import Union, Callable, Type
from collections import OrderedDict
from torch.utils.data.dataloader import DataLoader

from vortex.networks.models import create_model_components
from vortex.utils.data.dataset.wrapper import DatasetWrapper
from vortex.utils.data.collater import create_collater
from vortex.utils.logger.base_logger import ExperimentLogger
from vortex.utils.logger import create_logger
from vortex_runtime import model_runtime_map

__all__ = ['create_model','create_runtime_model','create_dataset','create_dataloader','create_experiment_logger','create_exporter']

def create_model(model_config : EasyDict,
                 state_dict : Union[str, dict, Path] = None, ## path to model or the actual state_dict
                 stage : str = 'train') -> EasyDict:
    if stage not in ['train','validate']:
        raise TypeError('Unknown model "stage" argument, got {}, expected "train" or "validate"'%stage)

    logging.info('Creating Pytorch model from experiment file')

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
    model_components = create_model_components(
        model_name,
        preprocess_args=preprocess_args,
        network_args=network_args,
        loss_args=loss_args,
        postprocess_args=postprocess_args,
        stage=stage)

    if 'init_state_dict' in model_config or state_dict is not None:
        if isinstance(state_dict, Path):
            state_dict = str(state_dict)

        model_path = None
        # Load state_dict from config if specified in experiment file
        if 'init_state_dict' in model_config and state_dict is None:
            logging.info("Loading state_dict from configuration file : {}".format(model_config.init_state_dict))
            model_path = model_config.init_state_dict
        # If specified using function's parameter, override the experiment config init_state_dict
        elif isinstance(state_dict, str):
            logging.info("Loading state_dict : {}".format(state_dict))
            model_path = state_dict

        if ('init_state_dict' in model_config and state_dict is None) or isinstance(state_dict, str):
            assert model_path
            ckpt = torch.load(model_path)
            state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        assert isinstance(state_dict, (OrderedDict, dict))
        model_components.network.load_state_dict(state_dict, strict=True) 

    return model_components

def create_runtime_model(model_path : Union[str, Path],
                         runtime: str, 
                         output_name=["output"], 
                         *args, 
                         **kwargs):

    model_type = Path(model_path).name.rsplit('.', 1)[1]
    runtime_map = model_runtime_map[model_type]
    if not runtime in runtime_map :
        raise RuntimeError("runtime {} not supported yet; available : {}".format(
            runtime, ', '.join(runtime_map.keys())
        ))
    Runtime = runtime_map[runtime]
    model = Runtime(model_path, output_name=output_name, *args, **kwargs)
    return model

def create_dataset(dataset_config : EasyDict,
                   preprocess_config : EasyDict,
                   stage : str):
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

    return DatasetWrapper(dataset=dataset, stage=stage, preprocess_args=preprocess_config,
                          augmentations=augmentations, dataset_args=dataset_args)

def create_dataloader(dataloader_config : EasyDict,
                      dataset_config : EasyDict, 
                      preprocess_config : EasyDict, 
                      stage : str,
                      collate_fn : Union[Callable,str,None] = None):

    dataloader_config = deepcopy(dataloader_config)
    dataset_config = deepcopy(dataset_config)
    preprocess_config = deepcopy(preprocess_config)

    dataset = create_dataset(dataset_config=dataset_config, preprocess_config=preprocess_config, 
                             stage=stage)
    if isinstance(collate_fn, str):
        try:
            collater_args = dataloader_config.collater.args
        except:
            collater_args = {}
        collater_args['dataformat'] = dataset.data_format
        collate_fn = create_collater(collate_fn, **collater_args)
    elif not (hasattr(collate_fn, '__call__') or collate_fn is None):
        raise TypeError("Unknown type of 'collate_fn', should be in the type of string, "
            "Callable, or None. Got {}".format(type(collate_fn)))

    if 'module' in dataloader_config:
        dataloader_module = dataloader_config.module
    elif 'dataloader' in dataloader_config:
        dataloader_module = dataloader_config.dataloader
    else:
        raise RuntimeError("Dataloader module in 'dataloader_config.module' is not set "
            "in dataloader_config ({}).".format(dataloader_config))
    dataloader_args = dataloader_config.args

    if not dataloader_module in ('DataLoader', 'PytorchDataLoader'):
        RuntimeError("Dataloader module '{}' is not supported, currently only 'PytorchDataLoader'")
    return DataLoader(dataset, collate_fn=collate_fn, **dataloader_args)

def create_experiment_logger(config : EasyDict):
    logger = config.logging
    experiment_logger = create_logger(logger,config)

    return experiment_logger

def create_exporter(config: Union[EasyDict,dict], 
                    experiment_name: str, 
                    image_size: int, 
                    output_directory: Union[Path,str]='.'):
    from vortex.exporter.onnx import OnnxExporter
    from vortex.exporter.torchscript import TorchScriptExporter
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