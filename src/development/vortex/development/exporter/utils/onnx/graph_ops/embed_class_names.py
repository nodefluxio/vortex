import logging
import onnx

from .helper import make_class_names
from .base_ops import GraphOpsBase
from typing import Dict

logger = logging.getLogger(__name__)

def embed_class_names(model : onnx.ModelProto, class_names : Dict[str,int]) -> onnx.ModelProto:
    """
    embed class_names to model as `Constants`
    """
    logger.info("embedding class_names to model")
    class_names_constants, class_names_value_info = make_class_names(class_names)
    for class_names_constant, value_info in zip(class_names_constants, class_names_value_info):
        model.graph.node.append(class_names_constant)
        model.graph.output.append(value_info)
    return model

class EmbedClassNames(GraphOpsBase):
    def __init__(self, class_names: Dict[str,int]):
        self.class_names = class_names
    
    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        return embed_class_names(model, **vars(self))