import onnx

from .helper import get_outputs

from onnx import helper


def create_batch_output_sequence(model : onnx.ModelProto, output_name : str='output', output_prefix : str='output_'):
    """
    """
    outputs = get_outputs(model)
    batch_outputs = list(filter(lambda x: x.name.startswith(output_prefix), outputs))
    print(batch_outputs)
    get_node_name = lambda x: x.name
    batch_output_names = list(map(get_node_name, batch_outputs))
    ## squeeze outputs before construct sequence
    ## TODO : read dim first
    # squeezed_output_nodes = []
    # squeezed_output_names = []
    # for output in batch_outputs :
    #     squeezed_output_name = 'squeezed_{}'.format(output.name)
    #     squeezed_output_nodes.append(helper.make_node(
    #         'Squeeze',
    #         inputs=[output.name], axes=[0],
    #         outputs=[squeezed_output_name],
    #     ))
    #     squeezed_output_names.append(squeezed_output_name)
    ## create sequence info and node
    sequence_value_info = helper.make_sequence_value_info(
        name=output_name, shape=None,
        elem_type=onnx.TensorProto.FLOAT,
    )
    # sequence_construct_node = helper.make_node(
    #     'SequenceConstruct',
    #     inputs=squeezed_output_names,
    #     outputs=[output_name]
    # )
    sequence_construct_node = helper.make_node(
        'SequenceConstruct',
        inputs=batch_output_names,
        outputs=[output_name]
    )
    # for node in squeezed_output_nodes :
    #     model.graph.node.append(node)
    model.graph.node.append(sequence_construct_node)
    model.graph.output.append(sequence_value_info)
    return model
