import pytest
import onnx
from functools import partial
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from vortex.runtime.onnx.graph_ops.helper import get_metadata_prop

from .dummy_model import dummy_model

# shorter version of registry
from vortex.development import ONNX_GRAPH_OPS as graph_ops

def test_embed_metadata():
    model = dummy_model()
    op = graph_ops.create_from_args('EmbedMetadata', key='test_key', value=4)
    model = op(model)
    prop = get_metadata_prop(model, 'test_key')
    assert prop is not None
    assert prop.key == 'test_key'
    assert prop.value == '4'
    # to use without instantiating, use apply
    model = dummy_model()
    embed = graph_ops.get('EmbedMetadata').apply
    model = embed(model, key='test_key', value=4)

def test_embed_metadata_failtest():
    model = dummy_model()
    op = graph_ops.create_from_args('EmbedMetadata',4,4)
    # key is not string
    with pytest.raises(TypeError):
        op(model)