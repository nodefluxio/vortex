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
        """Embed information to model, including 'output_format' and 'class_names'

        Args:
            model (onnx.ModelProto): model
            props (Dict[str,Any]): information, must include 'output_format' and 'class_names'

        Returns:
            onnx.ModelProto: model
        """
        for key, value in props.items():
            if key == 'output_format':
                model = EmbedOutputFormatMetadata.apply(model,value)
            elif key == 'class_names':
                model = EmbedClassNamesMetadata.apply(model,value)
            else:
                formatter = lambda x: str(x)
                model = EmbedMetadata.apply(model,key,value,formatter)
        return model
    
    @classmethod
    def parse(cls, model: onnx.ModelProto) -> Dict[str,Any]:
        """Extract output_format and class_names from model

        Args:
            model (onnx.ModelProto): model

        Returns:
            Dict[str,Any]: dictionary contains 'output_format' and 'class_names'
        """
        output_format = EmbedOutputFormatMetadata.parse(model)
        class_names = EmbedClassNamesMetadata.parse(model)
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