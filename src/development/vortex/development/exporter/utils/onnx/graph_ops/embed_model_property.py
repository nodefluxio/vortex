import logging
import onnx
import parse
from typing import Dict, Any
from .base_ops import GraphOpsBase
from .embed_metadata import EmbedMetadata
from .embed_class_names_metadata import EmbedClassNamesMetadata
from .embed_output_format_metadata import EmbedOutputFormatMetadata

logger = logging.getLogger(__name__)

class EmbedModelProperty(GraphOpsBase):
    def __init__(self, props):
        self.props = props
    
    @classmethod
    def apply(cls, model: onnx.ModelProto, props: Dict[str,Any]):
        for key, value in props.items():
            if key == 'output_format':
                model = EmbedOutputFormatMetadata.apply(model,value)
            elif key == 'class_names':
                model = EmbedClassNamesMetadata.apply(model,value)
            else:
                formatter = lambda x: str(x)
                model = EmbedMetadata.apply(model,key,value,formatter)
        return model
    
    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        return self.apply(model,**vars(self))