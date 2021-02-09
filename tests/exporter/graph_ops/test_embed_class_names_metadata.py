import pytest
import onnx
from functools import partial
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from vortex.development.exporter.utils.onnx.graph_ops.helper import get_metadata_prop

from .dummy_model import dummy_model

# shorter version of registry
from vortex.development import onnx_graph_ops as graph_ops

def test_embed_class_names():
    class_names = ['cat', 'dog']
    model = dummy_model()
    args = {'class_names': class_names}
    op = graph_ops.create_from_dict('EmbedClassNamesMetadata', args)
    model = op(model)
    # onnx.save(model, 'test.onnx')
    # note that parse is class method
    assert op.parse(model) == {0: 'cat', 1: 'dog'}
    # alternative:
    class_name_parser = graph_ops.get('EmbedClassNamesMetadata').parse
    assert class_name_parser(model) == {0: 'cat', 1: 'dog'}
