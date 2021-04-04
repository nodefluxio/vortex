import pytest
import onnx
from functools import partial
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

from .dummy_model import dummy_model

# shorter version of registry
from vortex.development import ONNX_GRAPH_OPS as graph_ops

def test_embed_class_names():
    class_names = ['cat', 'dog']
    model = dummy_model()
    args = {'class_names': class_names}
    # instantiate from dict
    op = graph_ops.create_from_dict('EmbedClassNamesMetadata', args)
    model = op(model)
    # onnx.save(model, 'test.onnx')
    # note that parse is class method
    assert op.parse(model) == {0: 'cat', 1: 'dog'}
    # alternative:
    parser = graph_ops.get('EmbedClassNamesMetadata').parse
    # parsed value is Dict[int,str]
    assert parser(model) == {0: 'cat', 1: 'dog'}

    # test for dict as input
    model = dummy_model()
    class_names = {0: 'cat', 1: 'dog'}
    op = graph_ops.create_from_args('EmbedClassNamesMetadata', class_names)
    model = op(model)
    # parser is not stateful
    assert parser(model) == {0: 'cat', 1: 'dog'}
