import onnx
import numpy as np

try :
    from .helper import make_constants, make_slice_value_info, get_Ops, get_inputs, get_outputs, replace_node
except ImportError :
    import os, sys, inspect
    from pathlib import Path
    repo_root = Path(__file__).parent
    sys.path.insert(0, str(repo_root))
    from helper import make_constants, make_slice_value_info, get_Ops, get_inputs, get_outputs, replace_node

from onnx import helper
from onnx import numpy_helper
from typing import Union
from pathlib import Path


supported_ops = [
    'nms'
]

def nms_iou_threshold_as_input(model : onnx.ModelProto, input_name : str='iou_threshold') :
    """
    """
    inputs = get_inputs(model)
    outputs = get_outputs(model)
    nms_op, ids = get_Ops(model, 'NonMaxSuppression')
    assert len(nms_op) >= 1
    ## TODO : make input if input does not exist in graph (?)
    assert any(input_name == inp.name for inp in inputs)
    iou_threshold_value_info = None
    for inp in inputs :
        if inp.name == input_name :
            iou_threshold_value_info = inp
    print('found {} NonMaxSuppression : {}'.format(len(nms_op), nms_op))
    for nms, idx in zip(nms_op,ids) :
        ## index 3 for 'iou_threshold', see : https://github.com/onnx/onnx/blob/master/docs/Operators.md#NonMaxSuppression
        nms.input[3] = input_name
        model = replace_node(model, idx, nms)
        nms_op, ids = get_Ops(model, 'NonMaxSuppression')
    print('found {} NonMaxSuppression : {}'.format(len(nms_op), nms_op))
    return model

def main(model_path, output) :
    model = onnx.load(model_path)
    model = nms_iou_threshold_as_input(model)
    onnx.save(model, model_path if output is None else output)

if __name__ == '__main__' :
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    main(args.model, args.output)