from vortex.core.factory import create_dataloader
from vortex.utils.data.dataset.wrapper.basic_wrapper import BasicDatasetWrapper
from easydict import EasyDict
import torch
import pytest

classification_config = EasyDict({
  'train': {
    'dataset': 'ImageFolder',
    'args': {
      'root': 'tests/test_dataset/train'
    },
  },
})

pytorch_loader = EasyDict({
    'dataloader': 'PytorchDataLoader',
    'args': {
      'num_workers': 1,
      'batch_size': 4,
      'shuffle': True,
    },
  })

dali_loader = EasyDict({
    'dataloader': 'DALIDataLoader',
    'args': {
      'device_id' : 0,
      'num_thread': 1,
      'batch_size': 4,
      'shuffle': True,
    },
  })

preprocess_args = EasyDict({
        'input_size' : 640,
        'input_normalization' : {
            'mean' : [0,0,0],
            'std' : [1, 1, 1],
            'scaler' : 1
        },
    })

def check_loader(dataloader):
    for data in dataloader:
        fetched_data = data
        break
    assert isinstance(fetched_data[0],torch.Tensor)
    assert len(fetched_data[0].shape)==4 # N,C,H,W
    assert fetched_data[0].shape[2] == preprocess_args.input_size # Assume square input
    assert isinstance(len(dataloader),int)
    assert hasattr(dataloader,'dataset')
    assert isinstance(dataloader.dataset,BasicDatasetWrapper)

@pytest.mark.parametrize("config", [classification_config])
def test_pytorch_loader(config):

    config.dataloader = pytorch_loader
    dataloader = create_dataloader(dataset_config=config,
                                   preprocess_config = preprocess_args,
                                   collate_fn=None)

    check_loader(dataloader)
    

@pytest.mark.parametrize("config", [classification_config])
def test_gpu_dali_loader(config):
    config.dataloader = dali_loader
    dataloader = create_dataloader(dataset_config=config,
                                   preprocess_config = preprocess_args,
                                   collate_fn=None)
    check_loader(dataloader)

