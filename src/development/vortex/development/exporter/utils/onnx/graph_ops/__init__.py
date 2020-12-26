from .base_ops import GraphOpsBase
from .create_batch_output_sequence import CreateBatchOutputSequence
from .embed_class_names import EmbedClassNames
from .embed_output_format import EmbedOutputFormat
from .nms_iou_threshold_as_input import IOUThresholdAsInput
from .symbolic_shape_infer import SymbolicShapeInfer

class OpsRegistry:
    def __init__(self, base_class=None):
        self.base_class = base_class
        self.ops = {}
    
    def add(self, op: type):
        assert isinstance(op,type), "expect a type"
        if self.base_class is not None:
            assert issubclass(op,self.base_class), f"expect {op} to be derived from {self.base_class}"
        self.ops[op.__name__] = op
    
    def get(self, op: str, *args, **kwargs):
        assert op in self.ops, f"op {op} not registered"
        op_type = self.ops[op]
        return op_type(*args, **kwargs)

supported_ops = OpsRegistry(base_class=GraphOpsBase)

def get_op(op: str, *args, **kwargs):
    return supported_ops.get(op, *args, **kwargs)

supported_ops.add(CreateBatchOutputSequence)
supported_ops.add(EmbedClassNames)
supported_ops.add(EmbedOutputFormat)
supported_ops.add(IOUThresholdAsInput)
supported_ops.add(SymbolicShapeInfer)