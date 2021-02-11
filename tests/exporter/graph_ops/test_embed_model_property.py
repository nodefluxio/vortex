import pytest
import onnx
from functools import partial
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from vortex.runtime.onnx.graph_ops.helper import get_metadata_prop
from vortex.runtime.onnx.graph_ops.check_metadata import check_metadata

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
        class_names=class_names,
        other_prop=1234
    )
    op = graph_ops.create_from_args('EmbedModelProperty', props=props)
    model = op(model)
    class_name_parser = graph_ops.get('EmbedClassNamesMetadata').parse
    assert class_name_parser(model) == {0: 'cat', 1: 'dog'}
    output_format_parser = graph_ops.get('EmbedOutputFormatMetadata').parse
    assert output_format_parser(model) == out_fmt

    # check that other property is also embedded to model
    # using helper fn get_metadata_prop
    other_prop = get_metadata_prop(model, 'other_prop')
    assert other_prop is not None
    assert other_prop.key == 'other_prop'
    assert other_prop.value == '1234'

    assert check_metadata(model)
    model = dummy_model()
    assert not check_metadata(model)