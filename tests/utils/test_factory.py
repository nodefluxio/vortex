import pytest

from copy import deepcopy
from easydict import EasyDict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from vortex.development.utils.factory import (
    create_dataloader,
    create_dataset,
    create_model,
    create_collater
)

## TODO:
# - create_model
# - create_dataset
# - create_dataloader

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
