"""
helper function to extract information from onnx IR
"""

import onnx
import numpy as np

from collections import OrderedDict
from onnx import TensorProto
from copy import copy, deepcopy
from typing import List, Dict

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

def get_output_format(model : onnx.ModelProto) -> Dict[str,Dict[str,List]] :
    outputs = get_outputs(model)
    output_names = [op.name for op in outputs]
    get_output_name = lambda node : node.output[0]
    constant_ops = get_Ops(model, 'Constant')
    constant_ops = filter(lambda op : get_output_name(op[0]) in output_names, zip(*constant_ops))
    constant_ops = list(constant_ops)
    output_axes = filter(lambda op : '_axis' in get_output_name(op[0]), constant_ops)
    output_indices = filter(lambda op : '_indices' in get_output_name(op[0]), constant_ops)
    output_axes, output_indices = list(output_axes), list(output_indices)

    ## get output name
    output_names = list(map(lambda op : get_output_name(op[0]).replace('_axis',''), output_axes))

    ## transform to pair
    get_tensor_proto = lambda node : node.attribute[0].t
    get_data = lambda node : get_tensor_proto(node).int64_data
    to_name_tensor_pair = lambda op : (get_output_name(op[0]), get_data(op[0]))
    output_axes = list(map(to_name_tensor_pair, output_axes))
    output_indices = list(map(to_name_tensor_pair, output_indices))
    output_format = {fmt : {} for fmt in output_names}
    for name, data in output_axes :
        name = name.replace('_axis', '')
        output_format[name].update(dict(axis=np.array(data)))
    for name, data in output_indices :
        name = name.replace('_indices', '')
        output_format[name].update(dict(indices=np.array(data)))
    return output_format

def get_class_names(model : onnx.ModelProto) -> List[str] :
    outputs = get_outputs(model)
    output_names = [op.name for op in outputs]
    get_output_name = lambda node : node.output[0]
    constant_ops = get_Ops(model, 'Constant')
    constant_ops = list(filter(lambda op : get_output_name(op[0]) in output_names, zip(*constant_ops)))
    class_names = list(filter(lambda op : get_output_name(op[0]).endswith('_label'), constant_ops))

    ## transform to pair
    get_tensor_proto = lambda node : node.attribute[0].t
    get_data = lambda node : get_tensor_proto(node).int64_data
    to_name_tensor_pair = lambda op : (get_output_name(op[0]), get_data(op[0]))
    class_names = list(map(to_name_tensor_pair, class_names))
    ## sort by index
    class_names = list(map(lambda x: x[0].replace('_label',''), sorted(class_names, key=lambda x: int(x[1][0]))))
    return class_names

def remove_nodes(model : onnx.ModelProto, ids : List[int]) :
    nodes = []
    new_model = deepcopy(model)
    del new_model.graph.node[:]
    for i, node in enumerate(model.graph.node) :
        if not i in ids :
            new_model.graph.node.append(node)
    return new_model

def get_input_specs(model : onnx.ModelProto) -> OrderedDict :
    inputs = get_inputs(model)
    input_names = [inp.name for inp in inputs]
    input_types = [TensorProto.DataType.Name(inp.type.tensor_type.elem_type).lower() for inp in inputs]
    input_shape = [inp.type.tensor_type.shape for inp in inputs]
    input_shape = list(map(lambda shape : [dim.dim_value for dim in shape.dim], input_shape))
    input_specs = []
    for name, dtype, shape in zip(input_names, input_types, input_shape) :
        dtype = dtype if dtype != 'float' else 'float32' ## explicit float32
        input_specs.append((
            name, dict(
                shape=shape,
                type=dtype
        )))
    return OrderedDict(input_specs)

def get_output_names(model : onnx.ModelProto, ignore_suffix=['_axis', '_indices', '_label']) -> List[str] :
    outputs = get_outputs(model)
    ignored = lambda x : any(suffix in x for suffix in ignore_suffix)
    output_names = [op.name for op in outputs if not ignored(op.name)]
    return output_names

## TODO : move to tests
if __name__=="__main__" :
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="path to onnx model")
    args = parser.parse_args()
    model = onnx.load(args.model)
    output_format = get_output_format(model)
    outputs = get_outputs(model)
    inputs = get_input_specs(model)
    outputs = get_output_names(model)
    print(output_format, inputs, outputs)