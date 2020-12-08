from easydict import EasyDict
from pathlib import Path

from vortex.development.utils.data.dataset import dataset
from vortex.development.core.factory import create_dataloader

import pytest
import torch


dataset.register_dvc_dataset("obj_det_landmarks", path=Path("tests/test_dataset"))

preprocess_args = EasyDict({
    'input_size' : 640,
    'input_normalization' : {
        'mean' : [.3,.1,.2],
        'std' : [.4, .1, .2],
        'scaler' : 255
    },
})

# Classification
class_dataset_config = EasyDict(
    {
        'train': {
            'name': 'ImageFolder',
            'args': {
                'root': 'tests/test_dataset/classification/train'
            },
        }
    }
)
class_dataset_config.collate_fn = None

# Obj Det with Landmark
lndmrks_dataset_config = EasyDict({
    'train': {
        'name': 'TestObjDetLandmarksDataset',
        'args': {
            'train': True
        },
    }
})
lndmrks_dataset_config.collate_fn = 'RetinaFaceCollate'

dali_loader = EasyDict({
    'module': 'DALIDataLoader',
    'args': {
        'device_id' : 0,
        'num_thread': 1,
        'batch_size': 1,
        'shuffle': False,
    },
})

pytorch_loader = EasyDict({
    'module': 'PytorchDataLoader',
    'args': {
        'batch_size': 4,
        'shuffle': False,
    },
})

transforms = [
    {'transform': 'HorizontalFlip','args':{'p':1}},
    {'transform': 'VerticalFlip','args':{'p':1}},
    {'transform': 'RandomBrightnessContrast','args':{'p':1,'brightness_limit' : .2,'contrast_limit' : .2}},
    {'transform': 'RandomJitter','args':{'p' : 1,'nDegree' : 2}},
    {'transform': 'RandomHueSaturationValue','args':{'p' :1,'hue_limit' : 20,'saturation_limit': .3,'value_limit': .3}},
    {'transform': 'RandomWater','args':{'p' :1}},
    {'transform': 'RandomRotate','args':{'p':1,'angle_limit':45}},
]

excpected_raise_error_transform = ['RandomWater','RandomRotate']


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU")
@pytest.mark.parametrize(
    ("dataset_config", "transform"), 
    [
        pytest.param(
            cfg, trfm,
            marks=pytest.mark.xfail(
                cfg == lndmrks_dataset_config and trfm['transform'] in excpected_raise_error_transform,
                reason="expected transform error in landmarks"
            ),
            id="{} {}".format(name, n)
        )
        for (cfg, name) in [(class_dataset_config, "classification"), (lndmrks_dataset_config, "landmarks")]
        for (n, trfm) in enumerate(transforms)
    ]
)
def test_dali(dataset_config, transform):
    augmentations = [EasyDict({'module' : 'nvidia_dali','args' : {'transforms': [transform]}})]
    dataset_config.train.augmentations = augmentations

    dataloader = create_dataloader(dataloader_config=dali_loader,
                                    dataset_config=dataset_config,
                                    preprocess_config = preprocess_args,
                                    collate_fn=dataset_config.collate_fn)

    data = next(iter(dataloader))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU")
def test_neg_dali_aug_and_pytorch_data_loader():
    augmentations = [EasyDict({'module' : 'nvidia_dali','args' : {'transforms': [transforms[0]]}})]
    class_dataset_config.train.augmentations = augmentations

    # Expect RuntimeError if nvidia dali augmentation module used with pytorch dataloader
    with pytest.raises(RuntimeError):
        dataloader = create_dataloader(dataloader_config=pytorch_loader,
                                        dataset_config=class_dataset_config,
                                        preprocess_config = preprocess_args,
                                        collate_fn=class_dataset_config.collate_fn)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU")
def test_dali_loader_with_no_dali_aug():
    augmentations = [EasyDict({'module' : 'albumentations','args' : {'transforms': [transforms[0]]}})]
    class_dataset_config.train.augmentations = augmentations

    dataloader = create_dataloader(dataloader_config=dali_loader,
                                    dataset_config=class_dataset_config,
                                    preprocess_config = preprocess_args,
                                    collate_fn=class_dataset_config.collate_fn)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU")
def test_dali_loader_with_additional_external_aug():
    augmentations = [EasyDict({'module' : 'nvidia_dali','args' : {'transforms': [transforms[0]]}}),
                     EasyDict({'module' : 'albumentations','args' : {'transforms': [transforms[0]]}})]
    class_dataset_config.train.augmentations = augmentations

    dataloader = create_dataloader(dataloader_config=dali_loader,
                                    dataset_config=class_dataset_config,
                                    preprocess_config = preprocess_args,
                                    collate_fn=class_dataset_config.collate_fn)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU")
def test_neg_dali_aug_not_the_first_aug():
    augmentations = [EasyDict({'module' : 'albumentations','args' : {'transforms': [transforms[0]]}}),
                     EasyDict({'module' : 'nvidia_dali','args' : {'transforms': [transforms[0]]}})]
    class_dataset_config.train.augmentations = augmentations

    # Expect RuntimeError if dali augmentations is not in the first order of the augmentation list
    with pytest.raises(RuntimeError):
        dataloader = create_dataloader(dataloader_config=dali_loader,
                                        dataset_config=class_dataset_config,
                                        preprocess_config = preprocess_args,
                                        collate_fn=class_dataset_config.collate_fn)
