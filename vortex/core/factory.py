import os
import sys
import logging
import torch

from copy import deepcopy
from pathlib import Path
from easydict import EasyDict
from typing import Union, Callable, Type
from collections import OrderedDict

from vortex.networks.models import create_model_components
from vortex.utils.data.loader import create_loader,wrapper_format
from vortex.utils.data.dataset.wrapper import BasicDatasetWrapper,DefaultDatasetWrapper
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
                   stage : str,
                   wrapper_format : str = 'default'):
    dataset_config = deepcopy(dataset_config)
    
    if stage == 'train' :
        dataset = dataset_config.train.dataset
        try:
            augmentations = dataset_config.train.augmentations
        except:
            augmentations = []
        dataset_args = dataset_config.train.args
    elif stage == 'validate':
        dataset = dataset_config.eval.dataset
        augmentations = []
        dataset_args = dataset_config.eval.args
    else:
        raise TypeError('Unknown dataset "stage" argument, got {}, expected "train" or "validate"'%stage)

    if wrapper_format=='default':
        return DefaultDatasetWrapper(dataset=dataset, stage=stage, preprocess_args=preprocess_config,
                          augmentations=augmentations, dataset_args=dataset_args)
    elif wrapper_format=='basic':
        return BasicDatasetWrapper(dataset=dataset, stage=stage, preprocess_args=preprocess_config,
                          augmentations=augmentations, dataset_args=dataset_args)
    else:
        raise RuntimeError('Unknown dataset `wrapper_format`, should be either "default" or "basic", got {} '.format(wrapper_format))

def create_dataloader(dataset_config : EasyDict, 
                      preprocess_config : EasyDict, 
                      stage : str = 'train',
                      collate_fn : Union[Callable,str,None] = None ):

    dataset_config = deepcopy(dataset_config)
    preprocess_config = deepcopy(preprocess_config)

    dataloader_module = dataset_config.dataloader.dataloader

    # For backward compatibility purpose
    if dataloader_module=='DataLoader':
        dataloader_module='PytorchDataLoader'
    dataloader_module_args = dataset_config.dataloader.args

    dataset = create_dataset(dataset_config=dataset_config, 
                             stage=stage, 
                             preprocess_config=preprocess_config,
                             wrapper_format=wrapper_format[dataloader_module])
    if isinstance(collate_fn,str):
        collater_args = {}
        try:
            collater_args = config.dataset.dataloader.collater.args
        except:
            pass
        collater_args['dataformat'] = dataset.data_format
        collate_fn = create_collater(collate_fn, **collater_args)
    elif hasattr(collate_fn,'__call__') or collate_fn is None:
        pass
    else :
        raise TypeError('Unknown type of "collate_fn", should be in the type of string, Callable, or None. Got {}'%type(collate_fn))
    
    dataloader = create_loader(dataloader_module,dataset,collate_fn = collate_fn, **dataloader_module_args)
    return dataloader

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