import sys
from pathlib import Path
proj_path = Path(__file__).parents[2]
sys.path.insert(0, str(proj_path.joinpath('src', 'development')))

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
                    'root' : str(proj_path)
                }
            }
        }
    )

    data = create_dataset(torch_dataset_conf, stage="train", preprocess_config=preprocess_args)
    assert isinstance(data.dataset[0][0], str), "ImageFolder expected to return input "\
        "of type 'str', got %s" % type(data.dataset[0][0])

def test_darknet_dataset():
    assert "DarknetDataset" in dataset.all_datasets['external']
    ## make sure that "base" dataset can be instantiated
    ## note that this test is dependent on external files
    args = dict(
        txt_path=str(proj_path/'tests/test_dataset/obj_det/train.txt'),
        img_root=str(proj_path/'tests/test_dataset/obj_det/images'),
        names=str(proj_path/'tests/test_dataset/obj_det/names.txt'),
    )
    base_dataset = dataset.get_base_dataset("DarknetDataset", args)
    assert len(base_dataset) == 5
    ## make sure that dataset can be wrapped
    preprocess_args = dict(
        input_size=640,
        input_normalization=dict(
            mean=[0.5,0.5,0.5],
            std=[0.5,0.5,0.5],
        )
    )
    dataset_conf = dict(
        train=dict(
            dataset="DarknetDataset",
            args=args,
        )
    )
    dataset_ = create_dataset(EasyDict(dataset_conf), stage="train", preprocess_config=EasyDict(preprocess_args))
    assert len(dataset_) == 5
    assert len(dataset_.class_names) == 20 ## VOC dataet
    img, label = dataset_[0]
    ## should return image with desired size
    assert all(lhs==rhs for lhs, rhs in zip(img.shape,[3,640,640]))

if __name__ == "__main__":
    test_dataset_register_dvc()
    test_create_dataset()
    test_torchvision_dataset()
