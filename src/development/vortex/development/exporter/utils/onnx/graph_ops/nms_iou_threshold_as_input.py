import logging
import onnx

from onnx import helper

from .helper import (
    get_Ops, get_inputs, get_outputs, replace_node
)


supported_ops = [
    'nms'
]
logger = logging.getLogger(__name__)


def nms_iou_threshold_as_input(model: onnx.ModelProto, input_name: str='iou_threshold', force_rewire=False) :
    """
    Given mode, find all existing NonMaxSuppression ops, make sure its iou_threshold is input.
    Args:
        model: onnx model proto in which its nms ops are to be made as input
        input_name: desired input name for iou_threshold input to nms, if not exists, an input with input_name will be added to the graph
        force_rewire: nms' iou_threshold will be rewired to input_name, if set to True.
    Return:
        model: updated model
    """
    inputs = get_inputs(model)
    outputs = get_outputs(model)
    nms_ops, ids = get_Ops(model, 'NonMaxSuppression')
    logger.info(f'found {len(nms_ops)} NonMaxSuppression ops: {nms_ops}')
    assert len(nms_ops) >= 1
    # scan all inputs in the graph, check if desired input (input_name) exists
    # by definition the name should be unique
    value_info = list(filter(lambda info: info.name == input_name, inputs))
    input_exists = len(value_info) >= 1
    if not input_exists:
        # create input info and add to graph
        logger.debug(f"creating {input_name} as NonMaxSuppression input")
        # input to iou_threshold for NonMaxSuppression is single element 1D tensor
        # see : https://github.com/onnx/onnx/blob/master/docs/Operators.md#NonMaxSuppression
        value_info = helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, [1])
        model.graph.input.append(value_info)
    else:
        # otherwise, input_name is already exists in the graph
        value_info = value_info[0]
    iou_threshold_names = list(map(lambda node: node.input[3],nms_ops))
    is_input = lambda name: any(name == inp.name for inp in inputs)
    for nms, idx, iou_name in zip(nms_ops, ids, iou_threshold_names):
        # if nms op at idx already has iou as input
        # and rewire is not forced, just skip
        if is_input(iou_name) and not force_rewire:
            continue
        # if multiple nms exists, they will share same input
        # index 3 for 'iou_threshold'
        nms.input[3] = value_info.name
        model = replace_node(model, idx, nms)
    nms_ops, ids = get_Ops(model, 'NonMaxSuppression')
    logger.info(f'updated {len(nms_ops)} NonMaxSuppression ops: {nms_ops}')
    return model
