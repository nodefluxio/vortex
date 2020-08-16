import numpy as np

from vortex.runtime.basic_runtime import BaseRuntime
from vortex.runtime.onnx.helper import get_output_format, get_input_specs, get_output_names, get_class_names

from pathlib import Path
from collections import OrderedDict
from typing import Union, List, Dict, Tuple, Any

class OnnxRuntime(BaseRuntime) :
    """
    Standardized runtime class for onnxruntime environment;
    """
    graph_optimization_level = {
        'disable_all' : 0,
        'enable_basic' : 1,
        'enable_extended' : 2,
        'disable' : 0,
        'basic' : 1,
        'extended' : 2,
    }
    execution_mode = {
        'sequential' : 0,
        'parallel' : 1,
    }
    def __init__(self, model : Union[str,Path], providers : Any, fallback : bool, input_name : str = 'input', output_name : Union[str,List[str]] = 'output', execution_mode : Union[str,int] = 'sequential', graph_optimization_level : Union[str,int] = 'basic') :
        import onnxruntime
        import onnx
        sess_options = onnxruntime.SessionOptions()
        if graph_optimization_level in OnnxRuntime.graph_optimization_level.keys() :
            graph_optimization_level = OnnxRuntime.graph_optimization_level[graph_optimization_level]
        elif graph_optimization_level in OnnxRuntime.graph_optimization_level.values() :
            pass
        else :
            raise ValueError("unsupported graph optimization level, supported : %s" %OnnxRuntime.graph_optimization_level)
        if execution_mode in OnnxRuntime.execution_mode.keys() : 
            execution_mode = OnnxRuntime.execution_mode[execution_mode]
        elif execution_mode in OnnxRuntime.execution_mode.values() :
            pass
        else :
            raise ValueError("unsupported execution mode, supported : %s" %OnnxRuntime.execution_mode)
        graph_optimization_level = onnxruntime.capi.onnxruntime_pybind11_state.GraphOptimizationLevel(graph_optimization_level)
        execution_mode = onnxruntime.capi.onnxruntime_pybind11_state.ExecutionMode(execution_mode)
        sess_options.graph_optimization_level = graph_optimization_level
        sess_options.execution_mode = execution_mode
        self.session = onnxruntime.InferenceSession(str(model),sess_options=sess_options,providers=providers)
        if not fallback : 
            import warnings
            warnings.warn("disabling onnx runtime fallback")
            self.session.disable_fallback()
        onnx_protobuf = onnx.load(model)
        output_format = get_output_format(onnx_protobuf)
        input_specs = get_input_specs(onnx_protobuf)
        output_names = get_output_names(onnx_protobuf)
        if not isinstance(output_name, list) :
            output_name = [output_name]
        assert all(name in output_names for name in output_name), \
            "graph doesn't have output : {}".format(output_name)
        class_names = get_class_names(onnx_protobuf)
        super(OnnxRuntime,self).__init__(
            input_specs=input_specs, 
            output_name=output_name, 
            output_format=output_format,
            class_names=class_names,
        )
        assert len(self.output_name) == 1
    
    @staticmethod
    def is_available() :
        try :
            import onnxruntime
            import onnx
            return True
        except ImportError :
            return False
        
    def predict(self, *args, **kwargs) -> np.ndarray :
        run_args = {name : value for name, value in zip(self.input_specs, args)}
        run_args = {**run_args, **kwargs}
        outputs = self.session.run(
            self.output_name, run_args
        )
        return outputs[0]

class OnnxRuntimeCpu(OnnxRuntime) :
    def __init__(self, model : Union[str,Path], fallback : bool = False, *args, **kwargs) :
        import onnxruntime
        if not 'CPUExecutionProvider' in onnxruntime.capi._pybind_state.get_available_providers() :
            if not fallback :
                raise ImportError("CPUExecutionProvider not available")
        super(OnnxRuntimeCpu,self).__init__(model,providers=['CPUExecutionProvider'],fallback=fallback,*args,**kwargs)
    
    @staticmethod
    def is_available() :
        try :
            import onnxruntime
            import onnx
            return 'CPUExecutionProvider' in onnxruntime.capi._pybind_state.get_available_providers()
        except ImportError :
            return False

class OnnxRuntimeCuda(OnnxRuntime) :
    def __init__(self, model : Union[str,Path], fallback : bool = False, *args, **kwargs) :
        import onnxruntime
        if not 'CUDAExecutionProvider' in onnxruntime.capi._pybind_state.get_available_providers() :
            if not fallback :
                raise ImportError("CUDAExecutionProvider not available")
        super(OnnxRuntimeCuda,self).__init__(model,providers=['CUDAExecutionProvider'],fallback=fallback,*args,**kwargs)

    @staticmethod
    def is_available() :
        try :
            import onnxruntime
            import onnx
            return 'CUDAExecutionProvider' in onnxruntime.capi._pybind_state.get_available_providers()
        except ImportError :
            return False

class OnnxRuntimeTensorRT(OnnxRuntime) :
    def __init__(self, model : Union[str,Path], fallback : bool = False, *args, **kwargs) :
        import onnxruntime
        if not 'TensorrtExecutionProvider' in onnxruntime.capi._pybind_state.get_available_providers() :
            if not fallback :
                raise ImportError("TensorrtExecutionProvider not available")
        super(OnnxRuntimeTensorRT,self).__init__(model,providers=['TensorrtExecutionProvider'],fallback=fallback,*args,**kwargs)

    @staticmethod
    def is_available() :
        try :
            import onnxruntime
            import onnx
            return 'TensorrtExecutionProvider' in onnxruntime.capi._pybind_state.get_available_providers()
        except ImportError :
            return False