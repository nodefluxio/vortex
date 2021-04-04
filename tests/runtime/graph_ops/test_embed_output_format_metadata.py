import pytest
import onnx
from functools import partial
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from vortex.runtime.onnx.graph_ops.helper import get_metadata_prop

from .dummy_model import dummy_model

# shorter version of registry
from vortex.development import ONNX_GRAPH_OPS as graph_ops

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

def test_embed_output_format_metadata():
    model = dummy_model()
    # simply map to string
    formatter = lambda x: str(x)
    op = graph_ops.create_from_args('EmbedOutputFormatMetadata', output_format=out_fmt)
    model = op(model)
    # onnx.save(model, 'testing.onnx')
    # parse should return dictionary same as input
    assert op.parse(model) == out_fmt