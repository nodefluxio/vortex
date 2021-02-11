from .base_ops import GraphOpsBase
from .create_batch_output_sequence import CreateBatchOutputSequence
from .embed_class_names import EmbedClassNames
from .embed_output_format import EmbedOutputFormat
from .nms_iou_threshold_as_input import IOUThresholdAsInput
from .symbolic_shape_infer import SymbolicShapeInfer
from vortex.runtime.onnx.graph_ops.embed_metadata import EmbedMetadata
from vortex.runtime.onnx.graph_ops.embed_output_format_metadata import EmbedOutputFormatMetadata
from vortex.runtime.onnx.graph_ops.embed_class_names_metadata import EmbedClassNamesMetadata
from vortex.runtime.onnx.graph_ops.embed_model_property import EmbedModelProperty
from vortex.development.utils.registry import Registry

GRAPH_OPS = Registry("graph_ops",base_class=GraphOpsBase)

GRAPH_OPS.add(CreateBatchOutputSequence)
GRAPH_OPS.add(EmbedClassNames)
GRAPH_OPS.add(EmbedOutputFormat)
GRAPH_OPS.add(IOUThresholdAsInput)
GRAPH_OPS.add(SymbolicShapeInfer)
