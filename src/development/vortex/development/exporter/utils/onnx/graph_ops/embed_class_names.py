import onnx
import numpy as np

from .helper import make_constants, make_slice_value_info, make_class_names

from onnx import helper
from onnx import numpy_helper
from typing import Union, Dict, List
from pathlib import Path


supported_ops = [
    'embed_class_names'
]

def embed_class_names(model : onnx.ModelProto, class_names : Dict[str,int]) -> onnx.ModelProto:
    """
    embed class_names to model as `Constants`
    """
    print("embedding class_names to model : {}".format(model.graph.name))
    class_names_constants, class_names_value_info = make_class_names(class_names)
    for class_names_constant, value_info in zip(class_names_constants, class_names_value_info) :
        model.graph.node.append(class_names_constant)
        model.graph.output.append(value_info)
    return model

def get_ops(*args, **kwargs) :
    return embed_class_names
