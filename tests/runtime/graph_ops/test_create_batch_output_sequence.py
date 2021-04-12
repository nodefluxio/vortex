import onnx
import numpy as np
from functools import partial
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from vortex.runtime.onnx.graph_ops.helper import get_Ops, get_outputs

# shorter version of registry
from vortex.development import ONNX_GRAPH_OPS as graph_ops
get_op = graph_ops.create_from_args

def dummy_model():
    # Create input info (ValueInfoProto)
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, None)
    B = helper.make_tensor_value_info('B', TensorProto.FLOAT, None)

    # Create output info (ValueInfoProto)
    output_0 = helper.make_tensor_value_info('output_0', TensorProto.FLOAT, None)
    output_1 = helper.make_tensor_value_info('output_1', TensorProto.FLOAT, None)

    # dummy node
    node_def1 = helper.make_node(
        'Gemm', # node name
        ['A', 'B'], # inputs
        ['output_0'], # outputs
    )

    # dummy node
    node_def2 = helper.make_node(
        'Gemm', # node name
        ['A', 'B'], # inputs
        ['output_1'], # outputs
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def1, node_def2],
        'test-model',
        [A, B],
        [output_0, output_1],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='testing')
    return model_def

def test_create_batch_output_sequence():
    """Combine multiple output to single input sequence (list)
    """
    model = dummy_model()
    op = get_op('CreateBatchOutputSequence')
    # helper lambda to extract nodeproto names
    f = lambda x: x.name

    # preconditions, no "output", nor SequenceConstruct
    outputs = list(map(f,get_outputs(model)))
    assert 'output' not in outputs
    seq_ops, seq_ids = get_Ops(model, "SequenceConstruct")
    assert len(seq_ops) == 0

    # actually run transform
    model = op(model)

    # postcondictions, "output" is constructed from "output_0" and "output_1"
    outputs = list(map(f,get_outputs(model)))
    assert 'output' in outputs
    seq_ops, seq_ids = get_Ops(model, "SequenceConstruct")
    assert len(seq_ops) == 1
    seq_op_inputs = seq_ops[0].input
    # exactly 2 for output_0 & output_1
    assert len(seq_op_inputs) == 2
    assert "output_0" in seq_op_inputs
    assert "output_1" in seq_op_inputs