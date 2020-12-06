import onnx
import numpy as np

from .helper import make_constants, make_slice_value_info

from onnx import helper


INT_MAX = 2**32
MAX_BOXES = INT_MAX

supported_ops = [
    'multiclass_nms'
]

def class_gather_node(model : onnx.ModelProto, class_id : int, detection_name : str='detections', n_detections_name : str='n_detections') :
    """
    >>> classes = detections[:,:,-1].squeeze()
    >>> classes.shape
    (n_class_detections,)
    >>> class_eq = np.equal(classes,n)
    >>> class_n = np.take(detections, class_eq, axis=1)
    (1,n_class_detections,6)
    """
    class_node_name = 'class_%s' %class_id
    class_id_const = make_constants(
        value=np.ndarray([class_id],dtype=np.int64),
        name='class_%s_const' %class_id
    )
    class_gather_slice_info = make_slice_value_info(
        starts=np.array([0,0,-1]),
        ends=np.array([MAX_BOXES,MAX_BOXES,-1]),
        axes=np.array([1,1,1]),
        steps=np.array([1,1,1]),
        starts_name='class_%s_gather_starts'%class_id,
        ends_name='class_%s_gather_ends'%class_id,
    )[:2] ## take starts info and ends info only
    class_n_slice_node = helper.make_node(
        'Slice',
        inputs=[
            detection_name,
            'class_%s_gather_starts' %class_id, 
            'class_%s_gather_ends' %class_id
        ],
        outputs=[
            'class_%s_slice' %class_id,
        ]
    )
    class_eq_node = helper.make_node(
        'Equal',
        inputs=[
            'class_%s_slice' %class_id, 
            'class_%s_const' %class_id
        ],
        outputs=[
            'class_%s_eq' %class_id
        ]
    )
    class_eq_value_info = helper.make_tensor_value_info(
        'class_%s_eq' %class_id,
        onnx.TensorProto.INT64,
        ['n_class_%s_detections' %class_id]
    )
    detection_class_node_value_info = helper.make_tensor_value_info(
        'detection_class_%s' %class_id,
        onnx.TensorProto.FLOAT,
        [1, "n_detection_class_%s" %class_id, -1]
    )
    detection_class_node = helper.make_node(
        'Gather',
        inputs=[
            detection_name, 
            'class_%s_eq'%class_id
        ],
        outputs=[
            'detection_class_%s' %class_id
        ],
        axis=1
    )
    model.graph.value_info.append(class_eq_value_info)
    model.graph.value_info.append(detection_class_node_value_info)
    for slice_info in class_gather_slice_info :
        model.graph.node.append(slice_info)
    model.graph.node.append(class_id_const)
    model.graph.node.append(class_n_slice_node)
    model.graph.node.append(class_eq_node)
    model.graph.node.append(detection_class_node)

def multiclass_nms(graph : onnx.ModelProto, n_classes : int, bboxes_name : str='bboxes', scores_name : str='scores', iou_threshold_name : str='iou_threshold', score_threshold_name : str='score_threshold', detections_name : str='detections', check_model=True):
    """
    add multiclass nms operations to given graph
    """
    print("performing multiclass nms operations on graph : %s" %graph.graph.name)
    for i in range(n_classes) :
        class_gather_node(graph, class_id=i)
    # nms_node = helper.make_node(
    #     'NonMaxSuppression',
    #     inputs=[
    #         bboxes_name, 
    #         scores_name,
    #         'max_output_boxes_per_class',
    #         iou_threshold_name,
    #         score_threshold_name
    #     ],
    #     outputs=[
    #         'selected_indices'
    #     ]
    # )
    # selected_indices = helper.make_tensor_value_info(
    #     'selected_indices', 
    #     onnx.TensorProto.INT64, ["n_detections", 3]
    # )
    # max_output_boxes_per_class = make_constants(
    #     value=np.array([MAX_BOXES],dtype=np.int64),
    #     name='max_output_boxes_per_class'
    # )
    # slice_info = make_slice_value_info(
    #     starts=np.array([0,2]),
    #     ends=np.array([MAX_BOXES,MAX_BOXES]),
    #     axes=np.array([1]),
    #     steps=np.array([1])
    # )[:2] ## take starts and ends only
    # detection_indices = helper.make_tensor_value_info(
    #     'detection_indices', 
    #     onnx.TensorProto.INT64, ["n_detections", 1]
    # )
    # detection_indices_node = helper.make_node(
    #     'Slice',
    #     inputs=['selected_indices', 'starts', 'ends'],
    #     outputs=['detection_indices'],
    # )
    # squeezed_detection_indices = helper.make_tensor_value_info(
    #     'squeezed_detection_indices',
    #     onnx.TensorProto.INT64, ["n_detections"]
    # )
    # detection_indices_squeeze_node = helper.make_node(
    #     'Squeeze',
    #     inputs=['detection_indices'],
    #     outputs=['squeezed_detection_indices'],
    #     axes=[1]
    # )
    # gather_node = helper.make_node(
    #     'Gather',
    #     inputs=[detections_name, 'squeezed_detection_indices'],
    #     outputs=['output'],
    #     axis=1
    # )
    # output_value_info = helper.make_tensor_value_info(
    #     'output', 
    #     onnx.TensorProto.FLOAT, 
    #     ["n_detections", -1, -1]
    # )
    # for info in slice_info :
    #     graph.graph.node.append(info)
    # additional_outputs = [
    #     selected_indices, 
    #     output_value_info, 
    #     detection_indices, 
    #     squeezed_detection_indices
    # ]
    # additional_nodes = [
    #     max_output_boxes_per_class,
    #     nms_node,
    #     gather_node,
    #     detection_indices_node,
    #     detection_indices_squeeze_node,
    # ]
    # for output in additional_outputs :
    #     graph.graph.output.append(output)
    # for node in additional_nodes :
    #     graph.graph.node.append(node)
    if check_model :
        print("performing nms operations on graph : %s CHECKING MODEL" %graph.graph.name)
        onnx.checker.check_model(graph)
    print("performing nms operations on graph : %s DONE" %graph.graph.name)
    return graph

def get_ops(*args, **kwargs) :
    return multiclass_nms