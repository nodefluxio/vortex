from vortex.runtime.onnx.onnxruntime import OnnxRuntimeCpu,OnnxRuntimeCuda,OnnxRuntimeTensorRT
from vortex.runtime.torchscript import TorchScriptRuntimeCpu,TorchScriptRuntimeCuda

__all__ = ['check_available_runtime']

model_runtime_map = {
    'onnx': {
        'cpu': OnnxRuntimeCpu,
        'cuda': OnnxRuntimeCuda,
        'tensorrt': OnnxRuntimeTensorRT,
    },
    'pt': {
        'cpu': TorchScriptRuntimeCpu,
        'cuda': TorchScriptRuntimeCuda,
    },
}

def check_available_runtime() -> dict:
    """Function to check current environment's available runtime

    Returns:
        dict: Dictionary containing status of available runtime
    
    Example:
        ```python
        from vortex.runtime import check_available_runtime

        available_runtime = check_available_runtime()
        print(available_runtime)
        ```
    """
    result = dict()
    for name, runtime_map in model_runtime_map.items() :
        if name =='pt':
            name = 'torchscript'
        result[name]=dict()
        for rt in runtime_map:
            print('Runtime {} <{}>: {}'.format(
                name, runtime_map[rt].__name__, 'available' \
                    if runtime_map[rt].is_available() else 'unavailable'
            ))
            result[name][runtime_map[rt].__name__]= 'available' if runtime_map[rt].is_available()\
                    else 'unavailable'
    return result