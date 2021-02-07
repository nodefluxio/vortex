from .base_ops import GraphOpsBase
from .create_batch_output_sequence import CreateBatchOutputSequence
from .embed_class_names import EmbedClassNames
from .embed_output_format import EmbedOutputFormat
from .nms_iou_threshold_as_input import IOUThresholdAsInput
from .symbolic_shape_infer import SymbolicShapeInfer
from vortex.development.utils.registry import Registry

# TODO: naming fix
graph_ops_registry = Registry("graph_ops",base_class=GraphOpsBase)

graph_ops_registry.add(CreateBatchOutputSequence)
graph_ops_registry.add(EmbedClassNames)
graph_ops_registry.add(EmbedOutputFormat)
graph_ops_registry.add(IOUThresholdAsInput)
graph_ops_registry.add(SymbolicShapeInfer)