import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2].joinpath('src', 'development')))

from vortex.development.core.factory import create_model
from easydict import EasyDict


def test_create_softmax_model():
    softmax_model_conf = EasyDict({
        'name' : 'softmax',
        'network_args' : {
            'backbone' : 'shufflenetv2_x1.0',
            'n_classes' : 10,
            'pretrained_backbone': True,
            'freeze_backbone': False
        },
        'preprocess_args': {
            'input_size': 32,
            'input_normalization': {
                'mean': [0.4914, 0.4822, 0.4465],
                'std': [0.2023, 0.1994, 0.2010]
            }
        },
        'loss_args': {
            'reduction': 'mean'
        },
        'postprocess_args': {}
    })
    softmax_model_components=create_model(model_config=softmax_model_conf,stage='train')
    assert isinstance(softmax_model_components,EasyDict)
    assert 'network' in softmax_model_components.keys()
    assert hasattr(softmax_model_components.network,'task')
    assert hasattr(softmax_model_components.network,'output_format')
    assert 'postprocess' in softmax_model_components.keys()
    assert 'preprocess' in softmax_model_components.keys()
    assert 'loss' in softmax_model_components.keys()
    assert 'collate_fn' in softmax_model_components.keys()
    softmax_model_components=create_model(model_config=softmax_model_conf,stage='validate')
    assert isinstance(softmax_model_components,EasyDict)
    assert 'network' in softmax_model_components.keys()
    assert hasattr(softmax_model_components.network,'task')
    assert hasattr(softmax_model_components.network,'output_format')
    assert 'postprocess' in softmax_model_components.keys()
    assert 'preprocess' in softmax_model_components.keys()