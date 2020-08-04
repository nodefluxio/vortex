from vortex.core.factory import create_runtime_model
import numpy as np
import cv2

model_path = 'tests/output_test/test_classification_pipelines/test_classification_pipelines.onnx'

runtime_model = create_runtime_model(
    model_path = model_path,
    runtime = 'cpu'
)

print(type(runtime_model))

## Get model's input specifications and additional inferencing parameters

print(runtime_model.input_specs)

# Inferencing example

input_shape = runtime_model.input_specs['input']['shape']
batch_imgs = np.array([cv2.resize(cv2.imread('tests/images/cat.jpg'),(input_shape[2],input_shape[1]))])

## Make sure the shape of input data is equal to input specifications
assert batch_imgs.shape == tuple(input_shape)

## Additional parameters can be inspected from input_specs,
## E.g. `score_threshold` or `iou_threshold` for object detection
additional_input_parameters = {}

## Inference is done by utilizing __call__ method
prediction_results = runtime_model(batch_imgs,
                                    **additional_input_parameters)

print(prediction_results)