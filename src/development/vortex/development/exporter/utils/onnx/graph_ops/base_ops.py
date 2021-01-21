import abc
import onnx

class GraphOpsBase(metaclass=abc.ABCMeta):
    def __init__(self):
        """Base Class to represents onnx graph ops
        """
        pass

    @abc.abstractmethod
    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        pass
    
    def __call__(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Transform model, call derived run function.

        Args:
            model (onnx.ModelProto): model to be transformed

        Returns:
            onnx.ModelProto: transformed model
        """        
        transformed_model = self.run(model)
        assert isinstance(transformed_model, onnx.ModelProto)
        return transformed_model
