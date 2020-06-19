import onnx
import numpy as np

from .helper import make_constants, make_slice_value_info

from onnx import helper
from onnx import numpy_helper
from typing import Union
from pathlib import Path


supported_ops = [
    'nms'
]


INT_MAX = 2**32
MAX_BOXES = INT_MAX


def nms(graph : onnx.ModelProto, bboxes_name : str='bboxes', scores_name : str='scores', iou_threshold_name : str='iou_threshold', score_threshold_name : str='score_threshold', detections_name : str='detections', check_model=True):
    """
    add nms operations to given graph
    """
    print("performing nms operations on graph : %s" %graph.graph.name)
    nms_node = helper.make_node(
        'NonMaxSuppression',
        inputs=[bboxes_name, scores_name,'max_output_boxes_per_class',iou_threshold_name,score_threshold_name],
        outputs=['selected_indices']
    )
    selected_indices = helper.make_tensor_value_info(
        'selected_indices', 
        onnx.TensorProto.INT64, ["n_detections", 3]
    )
    max_output_boxes_per_class = make_constants(
        value=np.array([MAX_BOXES],dtype=np.int64),
        name='max_output_boxes_per_class'
    )
    slice_info = make_slice_value_info(
        starts=np.array([0,2]),
        ends=np.array([MAX_BOXES,MAX_BOXES]),
        axes=np.array([1]),
        steps=np.array([1])
    )[:2] ## take starts and ends only
    detection_indices = helper.make_tensor_value_info(
        'detection_indices', 
        onnx.TensorProto.INT64, ["n_detections", 1]
    )
    ## TODO : rename `starts` and `ends` input
    detection_indices_node = helper.make_node(
        'Slice',
        inputs=['selected_indices', 'starts', 'ends'],
        outputs=['detection_indices'],
    )
    squeezed_detection_indices = helper.make_tensor_value_info(
        'squeezed_detection_indices',
        onnx.TensorProto.INT64, ["n_detections"]
    )
    detection_indices_squeeze_node = helper.make_node(
        'Squeeze',
        inputs=['detection_indices'],
        outputs=['squeezed_detection_indices'],
        axes=[1]
    )
    gather_node = helper.make_node(
        'Gather',
        inputs=[detections_name, 'squeezed_detection_indices'],
        outputs=['output'],
        axis=1
    )
    output_value_info = helper.make_tensor_value_info(
        'output', 
        onnx.TensorProto.FLOAT, 
        ["n_detections", -1, -1]
    )
    for info in slice_info :
        graph.graph.node.append(info)
    additional_outputs = [
        selected_indices, 
        output_value_info, 
        detection_indices, 
        squeezed_detection_indices
    ]
    additional_nodes = [
        max_output_boxes_per_class,
        nms_node,
        gather_node,
        detection_indices_node,
        detection_indices_squeeze_node,
    ]
    for output in additional_outputs :
        graph.graph.output.append(output)
    for node in additional_nodes :
        graph.graph.node.append(node)
    if check_model :
        print("performing nms operations on graph : %s CHECKING MODEL" %graph.graph.name)
        onnx.checker.check_model(graph)
    print("performing nms operations on graph : %s DONE" %graph.graph.name)
    return graph


def get_ops(*args, **kwargs) :
    return nms