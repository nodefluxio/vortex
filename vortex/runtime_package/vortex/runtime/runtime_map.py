from vortex.runtime.onnx.onnxruntime import OnnxRuntimeCpu,OnnxRuntimeCuda,OnnxRuntimeTensorRT
from vortex.runtime.torchscript import TorchScriptRuntimeCpu,TorchScriptRuntimeCuda

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