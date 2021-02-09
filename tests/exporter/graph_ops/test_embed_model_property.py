import pytest
import onnx
from functools import partial
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from vortex.development.exporter.utils.onnx.graph_ops.helper import get_metadata_prop

from .dummy_model import dummy_model

# shorter version of registry
from vortex.development import onnx_graph_ops as graph_ops

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

class_names = ['cat', 'dog']

def test_embed_model_property():
    model = dummy_model()
    props = dict(
        output_format=out_fmt,
        class_names=class_names
    )
    op = graph_ops.create_from_args('EmbedModelProperty', props=props)
    model = op(model)
    class_name_parser = graph_ops.get('EmbedClassNamesMetadata').parse
    assert class_name_parser(model) == {0: 'cat', 1: 'dog'}
    output_format_parser = graph_ops.get('EmbedOutputFormatMetadata').parse
    assert output_format_parser(model) == out_fmt
