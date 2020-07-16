from vortex.core.factory import create_dataloader
from easydict import EasyDict
import torch

classification_config = EasyDict({
  'train': {
    'dataset': 'ImageFolder',
    'args': {
      'root': 'tests/test_dataset/train'
    },
  },
  'dataloader': {
    'dataloader': 'PytorchDataLoader',
    'args': {
      'num_workers': 1,
      'batch_size': 4,
      'shuffle': True,
    },
  },
})

preprocess_args = EasyDict({
        'input_size' : 640,
        'input_normalization' : {
            'mean' : [0,0,0],
            'std' : [1, 1, 1]
        },
        'scaler' : 1
    })

def test_classification_data():

    dataloader = create_dataloader(dataset_config=classification_config,
                                   preprocess_config = preprocess_args,
                                   collate_fn=None)

    for data in dataloader:
        fetched_data = data
        break

    assert isinstance(fetched_data[0],torch.Tensor)
    assert len(fetched_data[0].shape)==4 # N,C,H,W