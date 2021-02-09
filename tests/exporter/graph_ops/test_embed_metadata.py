import pytest
import onnx
from functools import partial
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from vortex.development.exporter.utils.onnx.graph_ops.helper import get_metadata_prop

from .dummy_model import dummy_model

# shorter version of registry
from vortex.development import onnx_graph_ops as graph_ops

def test_embed_metadata():
    model = dummy_model()
    # simply map to string
    formatter = lambda x: str(x)
    op = graph_ops.create_from_args('EmbedMetadata', key='test_key', value=4, formatter=formatter)
    model = op(model)
    prop = get_metadata_prop(model, 'test_key')
    assert prop is not None
    assert prop.key == 'test_key'
    assert prop.value == '4'

def test_embed_metadata_failtest():
    model = dummy_model()
    op = graph_ops.create_from_args('EmbedMetadata', 4,4,lambda x: x)
    with pytest.raises(TypeError):
        op(model)

    with pytest.raises(TypeError):
        identity = lambda x: x
        # run immutable function
        op.apply(model,4,4,identity)