import onnx
import numpy as np

from .helper import make_constants, make_slice_value_info, make_output_format

from onnx import helper
from onnx import numpy_helper
from typing import Union, Dict, List
from pathlib import Path


supported_ops = [
    'embed_output_format'
]

def embed_output_format(model : onnx.ModelProto, output_format : Dict[str,Union[List[int],int]]) -> onnx.ModelProto:
    """
    embed output_format to model as `Constants`
    """
    print("embedding output_format to model : {}".format(model.graph.name))
    output_format_constants, output_format_value_info = make_output_format(output_format)
    for output_format_constant, value_info in zip(output_format_constants, output_format_value_info) :
        model.graph.node.append(output_format_constant)
        model.graph.output.append(value_info)
    return model

def get_ops(*args, **kwargs) :
    return embed_output_format
