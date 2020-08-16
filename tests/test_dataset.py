

import os
import sys
import subprocess
from pathlib import Path
proj_path = os.path.abspath(Path(__file__).parents[1])
sys.path.append(proj_path)
sys.path.append('src/development')
from easydict import EasyDict

from vortex.development.utils.data.dataset import dataset
from vortex.development.core.factory import create_dataset

def test_dataset_register_dvc():
    dataset.register_dvc_dataset("dummy_dataset", path=Path("tests"))
    assert "DummyDataset" in dataset.all_datasets['external']

def test_create_dataset():
    preprocess_args = EasyDict({
        'input_size' : 640,
        'input_normalization' : {
            'mean' : [0.5, 0.5, 0.5],
            'std' : [0.5, 0.5, 0.5]
        }
    })

    dummy_dataset_conf = EasyDict(
        {
            'train' : {
                'dataset' : "DummyDataset",
                'args' : {
                    'msg' : 'this is just a dummy'
                }
            }
        }
    )

    dataset.register_dvc_dataset("dummy_dataset", path=Path("tests"))
    data = create_dataset(dummy_dataset_conf, stage="train", preprocess_config=preprocess_args)
    assert data.dataset.kwargs["msg"] == dummy_dataset_conf.train.args["msg"]

def test_torchvision_dataset():
    from torchvision.datasets.folder import IMG_EXTENSIONS
    IMG_EXTENSIONS += ('.py', ) ## workaround so it can predict .py

    preprocess_args = EasyDict({
        'input_size' : 640,
        'input_normalization' : {
            'mean' : [0.5, 0.5, 0.5],
            'std' : [0.5, 0.5, 0.5]
        }
    })

    torch_dataset_conf = EasyDict(
        {
            'train' : {
                'dataset' : "ImageFolder",
                'args' : {
                    'root' : proj_path
                }
            }
        }
    )

    data = create_dataset(torch_dataset_conf, stage="train", preprocess_config=preprocess_args)
    assert isinstance(data.dataset[0][0], str), "ImageFolder expected to return input "\
        "of type 'str', got %s" % type(data.dataset[0][0])


if __name__ == "__main__":
    test_dataset_register_dvc()
    test_create_dataset()
    test_torchvision_dataset()
