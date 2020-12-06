import numpy as np

from vortex.runtime.basic_runtime import BaseRuntime
from pathlib import Path
from typing import Union
from collections import OrderedDict

# __all__ = [
#     "TorchScriptRuntime", 
#     "TorchScriptRuntimeCpu", 
#     "TorchScriptRuntimeCuda"
# ]

class TorchScriptRuntime(BaseRuntime):
    def __init__(self, model: Union[str, Path], device: Union[str,None], 
                 *args, **kwargs):
        import torch
        import torchvision
        if isinstance(model, (str, Path)):
            if not str(model).endswith('.pt') or str(model).endswith('.pth'):
                raise RuntimeError("Unknown model file extension from {}".format(str(model)))
            model = torch.jit.load(str(model))
        elif not isinstance(model, torch.jit.ScriptModule):
            raise RuntimeError("Unknown model type of {}, TorchScriptRuntime only "\
                "accept model of type 'torch.jit.ScriptModule'".format(type(model)))
        if isinstance(device, str):
            if device == "gpu":
                device = "cuda"
            device = torch.device(device)
        self.device = device
        self.model = model.to(device).eval()

        output_field = [name.rsplit('_', 1)[0] for name,_ in model.named_buffers(recurse=False)
            if name.endswith(('axis', 'indices'))]
        output_format = {
            name: {key: getattr(model, name + "_" + key).tolist() for key in ("axis", "indices")} 
                for name in output_field
        }
        input_spec = OrderedDict([
            (name.replace("_input_shape", ""), {
                "shape": shape.tolist(), 
                "type": "uint8" if name == "input" else "float"
            }) 
            for name, shape in self.model.named_buffers(recurse=False) if name.endswith("_input_shape")
        ])
        class_names = [
            (name.replace('_label',''), int(idx)) 
            for name, idx in self.model.named_buffers(recurse=False) \
                if name.endswith('_label')
        ]
        class_names = list(map(lambda x: x[0], sorted(class_names, key=lambda x: x[1])))
        super(TorchScriptRuntime, self).__init__(
            input_specs=input_spec, 
            output_name="output", 
            output_format=output_format, 
            class_names=class_names,
        )
        self.input_pos = {
            name: getattr(self.model, name + '_input_pos').item() for name in input_spec.keys()
        }
        
    ## TODO : check signature properly (?)
    def predict(self, x, *args, **kwargs) -> np.ndarray:
        import torch
        ## TODO: use keyword arguments, currently the exporter doesn't support it
        # args = {name: torch.tensor(value) for name, value in zip(self.input_specs, args)}
        # kwargs = {name: torch.tensor(value) for name, value in kwargs.items()}
        # kwargs = {**args, **kwargs}

        args, kwargs = self._resolve_inputs(*args, **kwargs)
        with torch.no_grad():
            x = torch.as_tensor(x, device=self.device)
            output = self.model(x, *args, **kwargs)
        if isinstance(output, torch.Tensor):
            output = output.cpu().numpy()
        elif isinstance(output, tuple):
            output = list(out.cpu().numpy() for out in output)
        return output

    @staticmethod
    def is_available(device="cpu"):
        try:
            import torch
            if device == "gpu":
                device = "cuda"
            device = str(device)
            if device == "cpu":
                return True     # cpu runtime always available
            elif "cuda" in device:
                return torch.cuda.is_available()
            else:
                raise RuntimeError("Unknown device of '{}'".format(device))
        except:
            return False

    def _resolve_inputs(self, *args, **kwargs):
        import torch
        args = list(args)
        for name, val in kwargs.items():
            args.insert(self.input_pos[name]-1, torch.tensor(val, device=self.device))
        return tuple(args), {}


class TorchScriptRuntimeCpu(TorchScriptRuntime):
    def __init__(self, model: Union[str, Path], *args, **kwargs):
        super(TorchScriptRuntimeCpu, self).__init__(model, device="cpu")

    @staticmethod
    def is_available():
        return TorchScriptRuntime.is_available(device="cpu")

class TorchScriptRuntimeCuda(TorchScriptRuntime):
    def __init__(self, model: Union[str, Path], device_id: Union[int,None] = None,
                 *args, **kwargs):
        if not self.is_valid_device(device_id):
            raise RuntimeError("CUDA GPU device {} is not available".format(device_id))
        device = "cuda"
        if device_id is not None:
            device = device + ":{}".format(device_id)
        super(TorchScriptRuntimeCuda, self).__init__(model, device=device)

    @staticmethod
    def is_available(device_id: Union[int] = None):
        try:
            import torch
            cuda_available = TorchScriptRuntime.is_available(device=torch.device("cuda"))
            valid = TorchScriptRuntimeCuda.is_valid_device(device_id)
            return cuda_available and valid
        except:
            return False

    @staticmethod
    def is_valid_device(device_id: Union[int] = None):
        import torch
        if device_id is not None and device_id > 0:
            if device_id < 0 or device_id >= torch.cuda.device_count():
                return False
        return True
