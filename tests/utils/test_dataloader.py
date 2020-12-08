from vortex.development.core.factory import create_dataloader
from vortex.development.utils.data.dataset.wrapper.basic_wrapper import BasicDatasetWrapper

from easydict import EasyDict
import torch
import pytest

classification_config = EasyDict({
    'train': {
        'dataset': 'ImageFolder',
        'args': {
            'root': 'tests/test_dataset/classification/train'
        },
    },
})

pytorch_loader = EasyDict({
    'module': 'PytorchDataLoader',
    'args': {
      'num_workers': 1,
      'batch_size': 4,
      'shuffle': True,
    },
  })

dali_loader = EasyDict({
    'module': 'DALIDataLoader',
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

@pytest.mark.parametrize(
    "dataloader_cfg", [
        pytest.param(pytorch_loader, id="pytorch dataloader"), 
        pytest.param(
            dali_loader, id="dali dataloader",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU")
        )
    ]
)
def test_dataloader(dataloader_cfg):
    dataloader = create_dataloader(dataloader_config=dataloader_cfg,
                                   dataset_config=classification_config,
                                   preprocess_config = preprocess_args,
                                   collate_fn=None)
    fetched_data = next(iter(dataloader))
    assert isinstance(fetched_data[0], torch.Tensor)
    assert len(fetched_data[0].shape) == 4 # N,C,H,W
    assert fetched_data[0].shape[2] == preprocess_args.input_size # Assume square input
    assert isinstance(len(dataloader), int)
    assert hasattr(dataloader, 'dataset')
    assert isinstance(dataloader.dataset, BasicDatasetWrapper)
