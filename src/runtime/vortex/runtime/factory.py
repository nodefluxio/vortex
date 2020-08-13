from . import model_runtime_map
from .basic_runtime import BaseRuntime
from pathlib import Path
from typing import Union,Type

__all__=['create_runtime_model']

def create_runtime_model(model_path : Union[str, Path],
                         runtime: str, 
                         output_name=["output"], 
                         *args, 
                         **kwargs) -> Type[BaseRuntime]:
    """Functions to create runtime model

    Args:
        model_path (Union[str, Path]): Path to Intermediate Representation (IR) model file
        runtime (str): Backend runtime to be used, e.g. : 'cpu' or 'cuda' (Depends on available runtime options)
        output_name (list, optional): Runtime output(s) variable name. Defaults to ["output"].

    Raises:
        RuntimeError: Raises if selected `runtime` is not available

    Returns:
        Type[BaseRuntime]: Runtime model objects based on IR file model's type and selected `runtime`

    Example:
        ```python
        from vortex.runtime import create_runtime_model
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
        ```
    """

    model_type = Path(model_path).name.rsplit('.', 1)[1]
    runtime_map = model_runtime_map[model_type]
    if not runtime in runtime_map :
        raise RuntimeError("runtime {} not supported yet; available : {}".format(
            runtime, ', '.join(runtime_map.keys())
        ))
    Runtime = runtime_map[runtime]
    model = Runtime(model_path, output_name=output_name, *args, **kwargs)
    return model
