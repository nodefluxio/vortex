from vortex.runtime.onnx.graph_ops.base_ops import GraphOpsBase
from vortex.runtime.onnx.graph_ops.create_batch_output_sequence import CreateBatchOutputSequence
from vortex.runtime.onnx.graph_ops.nms_iou_threshold_as_input import IOUThresholdAsInput
from vortex.runtime.onnx.graph_ops.symbolic_shape_infer import SymbolicShapeInfer
from vortex.runtime.onnx.graph_ops.embed_metadata import EmbedMetadata
from vortex.runtime.onnx.graph_ops.embed_output_format_metadata import EmbedOutputFormatMetadata
from vortex.runtime.onnx.graph_ops.embed_class_names_metadata import EmbedClassNamesMetadata
from vortex.runtime.onnx.graph_ops.embed_model_property import EmbedModelProperty
from vortex.runtime.onnx.graph_ops.embed_metrics import EmbedMetrics
from vortex.development.utils.registry import Registry

GRAPH_OPS = Registry("graph_ops",base_class=GraphOpsBase)

GRAPH_OPS.add(CreateBatchOutputSequence)
GRAPH_OPS.add(IOUThresholdAsInput)
GRAPH_OPS.add(SymbolicShapeInfer)
GRAPH_OPS.add(EmbedMetadata)
GRAPH_OPS.add(EmbedOutputFormatMetadata)
GRAPH_OPS.add(EmbedClassNamesMetadata)
GRAPH_OPS.add(EmbedModelProperty)
GRAPH_OPS.add(EmbedMetrics)