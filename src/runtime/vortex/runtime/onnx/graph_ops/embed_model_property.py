import logging
import onnx
from typing import Dict, Any
from .base_ops import GraphOpsBase
from .embed_metadata import embed_metadata, parse_metadata
from .embed_class_names_metadata import embed_class_names_metadata, parse_class_names_metadata
from .embed_output_format_metadata import embed_output_format_metadata, parse_output_format_metadata

logger = logging.getLogger(__name__)

class EmbedModelProperty(GraphOpsBase):
    def __init__(self, props):
        self.props = props
    
    @classmethod
    def apply(cls, model: onnx.ModelProto, props: Dict[str,Any]):
        """Embed information to model, including 'output_format' and 'class_names'

        Args:
            model (onnx.ModelProto): model
            props (Dict[str,Any]): information, must include 'output_format' and 'class_names'

        Returns:
            onnx.ModelProto: model
        """
        for key, value in props.items():
            if key == 'output_format':
                model = embed_output_format_metadata(model,value)
            elif key == 'class_names':
                model = embed_class_names_metadata(model,value)
            else:
                model = embed_metadata(model,key,value)
        return model
    
    @classmethod
    def parse(cls, model: onnx.ModelProto) -> Dict[str,Any]:
        """Extract output_format and class_names from model

        Args:
            model (onnx.ModelProto): model

        Returns:
            Dict[str,Any]: dictionary contains 'output_format' and 'class_names'
        """
        output_format = parse_output_format_metadata(model)
        class_names = parse_class_names_metadata(model)
        return dict(output_format=output_format,class_names=class_names)
    
    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Run transformation

        Args:
            model (onnx.ModelProto): model

        Returns:
            onnx.ModelProto: model
        """
        return self.apply(model,**vars(self))

# alias
embed_model_property = EmbedModelProperty.apply

# alias
parse_model_property = EmbedModelProperty.parse