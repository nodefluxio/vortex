from vortex.core.factory import create_dataloader
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