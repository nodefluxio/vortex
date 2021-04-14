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
        """Embed information to existing model

        Args:
            model (onnx.ModelProto): model
            key (str): a name representing the value
            value (Any): value to be embedded into model
            formatter (Callable): a callable to format value to string

        Raises:
            TypeError: key is not string
            TypeError: return value from `formatter(value)`

        Returns:
            onnx.ModelProto: model
        """
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
        """Run transformation

        Args:
            model (onnx.ModelProto): model

        Returns:
            onnx.ModelProto: model with embedded metadata
        """
        return self.apply(model, **vars(self))

# alias
embed_metadata = EmbedMetadata.apply