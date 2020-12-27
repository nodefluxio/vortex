import logging
import onnx

from .helper import make_output_format

from typing import Union, Dict, List

logger = logging.getLogger(__name__)

def embed_output_format(model : onnx.ModelProto, output_format : Dict[str,Union[List[int],int]]) -> onnx.ModelProto:
    """
    embed output_format to model as `Constants`
    """
    logger.info("embedding output_format to model")
    output_format_constants, output_format_value_info = make_output_format(output_format)
    for output_format_constant, value_info in zip(output_format_constants, output_format_value_info):
        model.graph.node.append(output_format_constant)
        model.graph.output.append(value_info)
    return model

def get_ops(*args, **kwargs) :
    return embed_output_format

from .base_ops import GraphOpsBase

class EmbedOutputFormat(GraphOpsBase):
    def __init__(self, output_format: Dict[str,Union[List[int],int]]):
        self.output_format = output_format
    
    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        return embed_output_format(model, **vars(self))