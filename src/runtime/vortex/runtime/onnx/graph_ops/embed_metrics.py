import logging
import onnx
import parse
import json
from typing import Any, List, Union
from .helper import get_metadata_prop
from .base_ops import GraphOpsBase
from .embed_metadata import EmbedMetadata

logger = logging.getLogger(__name__)

class EmbedMetrics(GraphOpsBase):
    prefix = 'vortex.metrics'
    def __init__(self, metrics):
        self.metrics = metrics
    
    @classmethod
    def apply(cls, model: onnx.ModelProto, metrics: Union[List[Any],Any]) -> onnx.ModelProto:
        """Embed given metrics to model, the embedded metrics is represented by its typename.

        Args:
            model (onnx.ModelProto): model to be embedded with metrics
            metrics (Union[List[Any],Any]): metrics to be embedded

        Returns:
            onnx.ModelProto: model with embedded metrics
        """
        # embed metrics
        get_class_names = lambda x: type(x).__name__
        value = []
        if not isinstance(metrics, list):
            metrics = [metrics]
        for i, metric in enumerate(metrics):
            value.append(get_class_names(metric))
        args = dict(
            key=cls.prefix,
            value=value,
        )
        model = EmbedMetadata.apply(model, **args)
        return model
    
    @classmethod
    def parse(cls, model) -> Union[List,None]:
        """Given model, retrive embedded metrics

        Args:
            model (onnx.ModelProto): model with embedded metrics

        Returns:
            Union[List,None]: list of string representing metric names
        """
        metrics = get_metadata_prop(model, cls.prefix)
        if metrics is not None:
            # should be list of str
            metrics = str(metrics.value)
            metrics = json.loads(metrics)
        return metrics
    
    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        return self.apply(model, **vars(self))
