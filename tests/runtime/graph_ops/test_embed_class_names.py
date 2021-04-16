import onnx
from functools import partial
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from vortex.runtime.onnx.graph_ops.helper import get_Ops, get_class_names

# shorter version of registry
from vortex.development import ONNX_GRAPH_OPS as graph_ops
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

# common test var
class_names = dict(
    cat=0,
    dog=1,
)

def op_test(op, model):
    # actually transform model
    model = op._deprecated_run(model)

    constant_ops, constant_ids = get_Ops(model, 'Constant')
    assert len(constant_ops) == 2

    # helper lambda to check name
    find = lambda x, name: x.output[0] == name

    # check for cat_label, should be exactly one, no duplicate
    g = partial(find, name='cat_label')
    a = filter(g, constant_ops)
    assert len(list(a)) == 1

    # helper lambda to check name
    find = lambda x, name: x.output[0] == name

    # check for dog_label, should be exactly one, no duplicate
    g = partial(find, name='dog_label')
    a = filter(g, constant_ops)
    assert len(list(a)) == 1

    names = get_class_names(model)
    # note: get_class_names return lists
    assert names == ['cat', 'dog']

def test_embed_class_names():
    """Embed class names to graph
    """
    model = dummy_model()

    op = get_op('EmbedClassNamesMetadata', class_names=class_names)
    op_test(op, model)

def test_registry():
    assert "EmbedClassNamesMetadata" in graph_ops
    model = dummy_model()
    op = graph_ops.create_from_args('EmbedClassNamesMetadata', class_names=class_names)
    op_test(op, model)