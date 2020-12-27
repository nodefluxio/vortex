import onnx

class GraphOpsBase:
    def __init__(self):
        """Base Class to represents onnx graph ops
        """
        pass

    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        raise NotImplementedError
    
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
