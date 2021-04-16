import logging
import onnx
import inspect
import json
from typing import Union, Dict, List, Callable, Any
from .base_ops import GraphOpsBase
from .helper import get_metadata_prop

logger = logging.getLogger(__name__)

class EmbedMetadata(GraphOpsBase):
    def __init__(self, key: str, value: Any):
        self.key = key
        self.value = value
    
    @classmethod
    def apply(cls, model: onnx.ModelProto, key: str, value: Any) -> onnx.ModelProto:
        """Embed information to existing model

        Args:
            model (onnx.ModelProto): model
            key (str): a name representing the value
            value (Any): value to be embedded into model

        Raises:
            TypeError: key is not string
            TypeError: return value from `formatter(value)`

        Returns:
            onnx.ModelProto: model
        """
        str_value = json.dumps(value)
        if not isinstance(key, str):
            raise TypeError(f"expect key to be a string, got {type(key)}")
        prop = onnx.onnx_pb.StringStringEntryProto(key=key,value=str_value)
        # note that metadata_props is RepeatedField of StringStringEntryProto
        model.metadata_props.extend([prop])
        return model

    @classmethod
    def parse(cls, model: onnx.ModelProto, key: str):
        """Parse metadata stored at model using json

        Args:
            model (onnx.ModelProto): model to be extracted
            key (str): metadata key

        Returns:
            Any: json.loads result or None if key not exists
        """
        # value is maybe StringStringEntryProto
        value = get_metadata_prop(model, key)
        if value is not None:
            value = json.loads(str(value.value))
        return value

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

# alias
parse_metadata = EmbedMetadata.parse