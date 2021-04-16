import onnx
import numpy as np
from functools import partial
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from vortex.runtime.onnx.graph_ops.helper import get_Ops, get_inputs, make_constants

from vortex.development import ONNX_GRAPH_OPS as graph_ops
get_op = graph_ops.create_from_args

def dummy_model():
    boxes = helper.make_tensor_value_info('boxes', TensorProto.FLOAT, None)
    scores = helper.make_tensor_value_info('scores', TensorProto.FLOAT, None)
    max_output_boxes_per_class = helper.make_tensor_value_info('max_output_boxes_per_class', TensorProto.INT64, None)
    iou_threshold = np.array([0.9], np.float32)
    iou_threshold = make_constants(iou_threshold, 'iou_threshold')
    score_threshold = helper.make_tensor_value_info('score_threshold', AttributeProto.FLOAT, None)

    # Create one output (ValueInfoProto)
    selected_indices = helper.make_tensor_value_info('selected_indices', TensorProto.INT64, None)

    node_def = helper.make_node(
        'NonMaxSuppression', # node name
        ['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'], # inputs
        ['selected_indices'], # outputs
        center_point_box=0, # attributes
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def, iou_threshold],
        'test-model',
        [boxes, scores, max_output_boxes_per_class, score_threshold],
        [selected_indices],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='testing')
    return model_def

def test_iou_threshold_as_input():
    """Make "iou_threshold" constant as input, if any
    """
    model = dummy_model()
    op = get_op('IOUThresholdAsInput')
    f = lambda x: x.name

    # preconditions, iou_threshold is not an input
    inputs = list(map(f,get_inputs(model)))
    assert 'iou_threshold' not in inputs

    model = op(model)

    # postcondition, iou_threshold is an input
    inputs = list(map(f,get_inputs(model)))
    iou_inputs = list(filter(lambda x: x=='iou_threshold', inputs))
    assert 'iou_threshold' in inputs
    assert len(iou_inputs) == 1

    # note that this also works if iou_threshold already registered
    model = op(model)
    inputs = list(map(f,get_inputs(model)))
    iou_inputs = list(filter(lambda x: x=='iou_threshold', inputs))
    assert len(iou_inputs) == 1