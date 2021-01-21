import pytest
import torch
import pytorch_lightning as pl

from copy import deepcopy
from easydict import EasyDict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.hub import load_state_dict_from_url

from ..common import state_dict_is_equal

from vortex.development.utils.factory import (
    create_dataloader,
    create_dataset,
    create_model,
    create_collater
)
from vortex.development.utils.parser.parser import load_config
from vortex.development.networks.models import supported_models


## TODO: test create_dataset


DATASET_CFG = dict(
    train=dict(
        name='ImageFolder',
        args=dict(root='tests/test_dataset/classification/train')
    ),
    eval=dict(
        name='ImageFolder',
        args=dict(root='tests/test_dataset/classification/val')
    )
)
DATALOADER_CFG = dict(
    module="PytorchDataLoader",
    args=dict(num_workers=1, batch_size=2, shuffle=True),
)
MODEL_CFG = dict(
    preprocess_args=dict(
        input_size=224,
        input_normalization=dict(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    )
)


@pytest.mark.parametrize(
    ('stage', 'collate_fn'),
    [
        pytest.param('train', None, id='train stage'),
        pytest.param('validate', None, id='validate stage'),
        pytest.param('train', 'DETRCollate', id='train stage with collater'),
        pytest.param('invalid', None, id='invalid stage', marks=pytest.mark.xfail)
    ]
)
def test_create_dataloaders(stage, collate_fn):
    ## the same dataloader cfg for train and eval
    dataloader_cfg = EasyDict(deepcopy(DATALOADER_CFG))
    dataset_cfg = EasyDict(deepcopy(DATASET_CFG))
    preprocess_cfg = EasyDict(deepcopy(MODEL_CFG['preprocess_args']))

    dataloader = create_dataloader(dataloader_cfg, dataset_cfg, preprocess_cfg, stage, collate_fn)
    assert isinstance(dataloader, DataLoader)
    assert dataloader.num_workers == 1 and dataloader.batch_size == 2


    ## separate dataloader cfg for train and eval
    dataloader_cfg = EasyDict(dict(
        train=deepcopy(DATALOADER_CFG),
        eval=deepcopy(DATALOADER_CFG)
    ))
    dataloader_cfg['eval']['args']['shuffle'] = False
    dataloader = create_dataloader(dataloader_cfg, dataset_cfg, preprocess_cfg, stage, collate_fn)

    assert isinstance(dataloader, DataLoader)
    assert dataloader.num_workers == 1 and dataloader.batch_size == 2
    if stage == 'train':
        assert isinstance(dataloader.sampler, RandomSampler)
    elif stage == 'validate':
        assert isinstance(dataloader.sampler, SequentialSampler)


    ## invalid dataloader cfg
    with pytest.raises(RuntimeError):
        dataloader_cfg = EasyDict({
            'invalid': deepcopy(DATALOADER_CFG)
        })
        dataloader = create_dataloader(dataloader_cfg, dataset_cfg, preprocess_cfg, stage, collate_fn)


@pytest.mark.parametrize(
    "stage", ['train', 'validate']
)
def test_create_dataloader_callable_collate(stage):
    dataloader_cfg = EasyDict(deepcopy(DATALOADER_CFG))
    dataset_cfg = EasyDict(deepcopy(DATASET_CFG))
    preprocess_cfg = EasyDict(deepcopy(MODEL_CFG['preprocess_args']))

    collate_fn = create_collater('DETRCollate', dataformat=dict(class_label=None))

    dataloader = create_dataloader(dataloader_cfg, dataset_cfg, preprocess_cfg, stage, collate_fn)
    assert isinstance(dataloader, DataLoader)
    assert dataloader.num_workers == 1 and dataloader.batch_size == 2


def test_create_model(tmp_path):
    config_path = "tests/config/test_classification_pipelines.yml"
    pretrained_url = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
    config = EasyDict(load_config(config_path))
    config['output_directory'] = str(tmp_path)
    config['model']['network_args']['backbone'] = 'resnet18'

    config['model']['network_args']['pretrained_backbone'] = True
    base_model_pretrained = supported_models['softmax'](
        preprocess_args=config['model']['preprocess_args'],
        network_args=config['model']['network_args'],
        loss_args=config['model']['loss_args'],
        postprocess_args=config['model']['postprocess_args']
    )

    config['model']['network_args']['pretrained_backbone'] = False
    base_model = supported_models['softmax'](
        preprocess_args=config['model']['preprocess_args'],
        network_args=config['model']['network_args'],
        loss_args=config['model']['loss_args'],
        postprocess_args=config['model']['postprocess_args']
    )

    ckpt_state_dict_path = tmp_path.joinpath('ckpt_state_dict.pth')
    ckpt_state_dict = base_model_pretrained.state_dict()
    torch.save(ckpt_state_dict, ckpt_state_dict_path)
    ckpt_path = tmp_path.joinpath('ckpt.pth')
    ckpt = dict(state_dict=ckpt_state_dict)
    torch.save(ckpt, ckpt_path)

    ## using config.model, normal run
    model = create_model(config.model)
    assert isinstance(model, pl.LightningModule)
    state_dict_is_equal(model.state_dict(), base_model.state_dict())

    ## using config, normal run
    model = create_model(config)
    assert isinstance(model, pl.LightningModule)
    state_dict_is_equal(model.state_dict(), base_model.state_dict())

    ## using 'state_dict' path as 'str', with 'state_dict' key
    model = create_model(config.model, state_dict=str(ckpt_path))
    assert isinstance(model, pl.LightningModule)
    state_dict_is_equal(ckpt_state_dict, model.state_dict())

    ## using path as str, file is state dict saved directly
    model = create_model(config.model, state_dict=str(ckpt_state_dict_path))
    assert isinstance(model, pl.LightningModule)
    state_dict_is_equal(ckpt_state_dict, model.state_dict())

    ## using 'state_dict' path as 'Path'
    model = create_model(config.model, state_dict=ckpt_path)
    assert isinstance(model, pl.LightningModule)
    state_dict_is_equal(ckpt_state_dict, model.state_dict())

    ## using 'state_dict' from ckpt
    model = create_model(config.model, state_dict=ckpt_state_dict)
    assert isinstance(model, pl.LightningModule)
    state_dict_is_equal(ckpt_state_dict, model.state_dict())

    ## using 'state_dict' from ckpt, with 'state_dict' as key
    model = create_model(config.model, state_dict=ckpt)
    assert isinstance(model, pl.LightningModule)
    state_dict_is_equal(ckpt_state_dict, model.state_dict())

    ## using 'init_state_dict' from config
    new_cfg_model = deepcopy(config.model)
    new_cfg_model['init_state_dict'] = str(ckpt_state_dict_path)
    model = create_model(new_cfg_model)
    assert isinstance(model, pl.LightningModule)
    state_dict_is_equal(ckpt_state_dict, model.state_dict())

    ## using 'init_state_dict' from config, with 'state_dict' as key
    new_cfg_model = deepcopy(config.model)
    new_cfg_model['init_state_dict'] = str(ckpt_path)
    model = create_model(new_cfg_model)
    assert isinstance(model, pl.LightningModule)
    state_dict_is_equal(ckpt_state_dict, model.state_dict())

    ## if both 'init_state_dict' config and 'state_dict' argument is specified
    ## 'state_dict' argument should be of top priority
    new_cfg_model['init_state_dict'] = str(ckpt_path)
    model = create_model(new_cfg_model, state_dict=base_model.state_dict())
    assert isinstance(model, pl.LightningModule)
    state_dict_is_equal(base_model.state_dict(), model.state_dict())
