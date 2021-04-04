import pytest
import onnx
from functools import partial
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from vortex.runtime.onnx.graph_ops.helper import get_metadata_prop

import pytorch_lightning as pl
from .dummy_model import dummy_model

# shorter version of registry
from vortex.development import ONNX_GRAPH_OPS as graph_ops

def test_embed_metrics():
    model = dummy_model()
    metrics = [pl.metrics.Accuracy(), pl.metrics.F1()]
    # can embed multiple metrics
    op = graph_ops.create_from_args('EmbedMetrics', metrics)
    model = op(model)
    prop = get_metadata_prop(model, op.prefix) # prefix = 'vortex.metrics'
    assert prop is not None
    assert prop.key == op.prefix
    parsed = op.parse(model)
    assert len(parsed) == 2
    assert parsed == ['Accuracy', 'F1']

def test_embed_metric():
    model = dummy_model()
    metrics = pl.metrics.Accuracy()
    # can also embed single metric
    op = graph_ops.create_from_args('EmbedMetrics', metrics)
    model = op(model)
    prop = get_metadata_prop(model, op.prefix) # prefix = 'vortex.metrics'
    assert prop is not None
    assert prop.key == op.prefix
    parsed = op.parse(model)
    assert len(parsed) == 1
    assert parsed == ['Accuracy']

def test_embed_metric_none():
    model = dummy_model()
    op = graph_ops['EmbedMetrics']
    parsed = op.parse(model)
    assert parsed is None