import numpy as np
import onnx
from onnx import helper
from copy import copy, deepcopy

def get_metadata_prop(model: onnx.ModelProto, key: str) -> onnx.onnx_pb.StringStringEntryProto:
    for prop in model.metadata_props:
        if prop.key == key:
            return prop
    return None

def make_constants(value : np.ndarray, name : str) :
    """
    given numpy array and a name, create onnx `Constants` node
    """
    data_type = onnx.numpy_helper.mapping.NP_TYPE_TO_TENSOR_TYPE[value.dtype]
    constants = helper.make_node(
        'Constant',
        inputs=[],
        outputs=[name],
        value=helper.make_tensor(
            name='value',
            data_type=data_type,
            dims=value.shape,
            vals=value
        )
    )
    return constants

def make_class_names(class_names : dict) :
    """
    creates constants of class_names
    """
    class_names_constants, class_names_value_info = [], []
    for key, value in class_names.items() :
        class_idx = [value] if isinstance(value, int) else value
        class_idx = np.asarray(class_idx)
        class_name = '{}_label'.format(key)
        class_names_constants.append(
            make_constants(
                value=class_idx, 
                name=class_name
            )
        )
        class_names_value_info.append(
            helper.make_tensor_value_info(
                class_name,
                onnx.TensorProto.INT64,
                class_idx.shape
            )
        )
    return class_names_constants, class_names_value_info

def make_output_format(output_format : dict) :
    """
    creates constants of output_format
    """
    output_format_constants, output_format_value_info = [], []
    for key, value in output_format.items() :
        assert 'indices' and 'axis' in value
        indices, axis = value['indices'], value['axis']
        axis = [axis] if isinstance(axis, int) else axis
        indices, axis = np.asarray(indices), np.asarray(axis)
        indices_name = '{}_indices'.format(key)
        axis_name = '{}_axis'.format(key)
        output_format_constants.append(
            make_constants(
                value=indices, 
                name=indices_name
            )
        )
        output_format_constants.append(
            make_constants(
                value=axis, 
                name=axis_name
            )
        )
        output_format_value_info.append(
            helper.make_tensor_value_info(
                indices_name,
                onnx.TensorProto.INT64,
                indices.shape
            )
        )
        output_format_value_info.append(
            helper.make_tensor_value_info(
                axis_name,
                onnx.TensorProto.INT64,
                axis.shape
            )
        )
    return output_format_constants, output_format_value_info

def get_inputs(model : onnx.ModelProto) :
    inputs = []
    for node in model.graph.input :
        inputs.append(copy(node))
    return inputs

def get_outputs(model : onnx.ModelProto) :
    outputs = []
    for node in model.graph.output :
        outputs.append(copy(node))
    return outputs

def get_Ops(model : onnx.ModelProto, op_type : str) -> list:
    ops, ids = [], []
    for id, node in enumerate(model.graph.node) :
        if node.op_type == op_type :
            ops.append(copy(node))
            ids.append(id)
    return ops, ids

def replace_node(model : onnx.ModelProto, id : int, new_node : onnx.NodeProto) :
    nodes = []
    new_model = deepcopy(model)
    del new_model.graph.node[:]
    for i, node in enumerate(model.graph.node) :
        new_model.graph.node.append(node if i!=id else new_node)
    return new_model