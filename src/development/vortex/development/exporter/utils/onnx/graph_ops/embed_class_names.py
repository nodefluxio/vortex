import logging
import onnx

from .helper import make_class_names

from typing import Dict

logger = logging.getLogger(__name__)

supported_ops = [
    'embed_class_names'
]

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

def get_ops(*args, **kwargs) :
    return embed_class_names
