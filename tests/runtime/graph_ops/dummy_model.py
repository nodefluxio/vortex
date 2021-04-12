import pytest
import onnx
from functools import partial
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

def dummy_model():
    # Create two input information (ValueInfoProto)
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, None)
    B = helper.make_tensor_value_info('B', TensorProto.FLOAT, None)

    # Create one output (ValueInfoProto)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, None)

    # dummy gemm node
    node_def = helper.make_node(
        'Gemm', # node name
        ['A', 'B'], # inputs
        ['output'], # outputs
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [A, B],
        [output],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='testing')
    return model_def