from vortex.core.factory import create_exporter
from easydict import EasyDict

export_config = EasyDict({
    'module': 'onnx',
    'args': {
        'opset_version' : 11,
    },
})

exporter = create_exporter(
    config=export_config,
    experiment_name='test_onnx',
    image_size=224,
    output_directory='.'
)

## Example exporting model

status = exporter(
    predictor=self.predictor,
    class_names=self.class_names,
    example_image_path=example_input
)