import logging
import onnx
import inspect
from typing import Union, Dict, List, Callable, Any
from .base_ops import GraphOpsBase

logger = logging.getLogger(__name__)

class EmbedMetadata(GraphOpsBase):
    def __init__(self, key: str, value: Any, formatter: Callable):
        self.key = key
        self.value = value
        self.formatter = formatter
    
    @classmethod
    def apply(cls, model: onnx.ModelProto, key: str, value: Any, formatter: Callable) -> onnx.ModelProto:
        str_value = formatter(value)
        if not isinstance(key, str):
            raise TypeError(f"expect key to be a string, got {type(key)}")
        if not isinstance(str_value, str):
            raise TypeError(f"expect formatter to return string, got {type(str_value)}")
        prop = onnx.onnx_pb.StringStringEntryProto(key=key,value=str_value)
        # note that metadata_props is RepeatedField of StringStringEntryProto
        model.metadata_props.extend([prop])
        return model

    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        return self.apply(model, **vars(self))