import pytest
import onnx
from functools import partial
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from vortex.development.exporter.utils.onnx.graph_ops.helper import get_Ops
from vortex.runtime.onnx.helper import get_output_format

# shorter version of registry
from vortex.development import onnx_graph_ops as graph_ops
get_op = graph_ops.create_from_args

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

out_fmt = dict(
    class_label=dict(
        indices=[0],
        axis=0,
    ),
    class_confidence=dict(
        indices=[1],
        axis=0,
    )
)

def op_test(op, model, out_fmt):
    model = op(model)
    # for visualization
    # onnx.save(model, 'testing.onnx')
    constant_ops, constant_ids = get_Ops(model, 'Constant')

    assert len(constant_ops) == 4

    # helper lambda to check name
    find = lambda x, name: x.output[0] == name

    # check for class_label_indices, should be exactly one, no duplicate
    g = partial(find, name='class_label_indices')
    a = filter(g, constant_ops)
    assert len(list(a)) == 1

    # check for class_label_axis, should be exactly one, no duplicate
    g = partial(find, name='class_label_axis')
    a = filter(g, constant_ops)
    assert len(list(a)) == 1

    # check for class_confidence_indices, should be exactly one, no duplicate
    g = partial(find, name='class_confidence_indices')
    a = filter(g, constant_ops)
    assert len(list(a)) == 1

    # check for class_confidence_axis, should be exactly one, no duplicate
    g = partial(find, name='class_confidence_axis')
    a = filter(g, constant_ops)
    assert len(list(a)) == 1

    # make sure output_format can be retrieved from model
    fmt = get_output_format(model)
    assert out_fmt == fmt

def test_embed_output_format():
    """Make sure embed_output_format graph ops running and the result can be retrieved correctly
    """
    model = dummy_model()

    with pytest.raises(AssertionError):
        # this should fail
        op = get_op('EmbedOutputFormat', output_format=out_fmt['class_label'])
        model = op(model)

    op = get_op('EmbedOutputFormat', output_format=out_fmt)
    op_test(op, model, out_fmt)

def test_registry():
    assert "EmbedOutputFormat" in graph_ops
    model = dummy_model()
    op = graph_ops.create_from_args('EmbedOutputFormat', output_format=out_fmt)
    op_test(op, model, out_fmt)

if __name__=='__main__':
    model = dummy_model()
    onnx.save(model, 'testing.onnx')
