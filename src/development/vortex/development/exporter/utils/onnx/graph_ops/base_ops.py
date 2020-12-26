import onnx

class GraphOpsBase:
    def __init__(self):
        pass

    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        raise NotImplementedError
    
    def __call__(self, model: onnx.ModelProto) -> onnx.ModelProto:
        transformed_model = self.run(model)
        assert isinstance(transformed_model, onnx.ModelProto)
        return transformed_model
