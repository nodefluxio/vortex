from vortex.core.factory import create_model
from easydict import EasyDict

model_config = EasyDict({
    'name': 'softmax',
    'network_args': {
        'backbone': 'efficientnet_b0',
        'n_classes': 10,
        'pretrained_backbone': True,
    },
    'preprocess_args': {
        'input_size': 32,
        'input_normalization': {
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2023, 0.1994, 0.2010],
        'scaler': 255,
        }
    },
    'loss_args': {
        'reduction': 'mean'
    }
})

model_components = create_model(
    model_config = model_config
)
print(model_components.keys())