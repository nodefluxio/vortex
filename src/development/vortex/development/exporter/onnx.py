import onnx
import torch
import enforce
import sys

from vortex.development.networks.modules.postprocess.base_postprocess import BasicNMSPostProcess, BatchedNMSPostProcess
from .utils.onnx.graph_ops import GRAPH_OPS
from .utils.onnx.graph_ops.helper import get_Ops

from typing import Union, List, Tuple, Any
from pathlib import Path

__all__ = [
    'export',
]

@enforce.runtime_validation
def export(model : Any, example_input : Union[Tuple[torch.Tensor,...],torch.Tensor], filename : str, input_names : List[str], output_names : List[str], opset_version : int = 9, *args, **kwargs) :
    torch.onnx.export(model, example_input, filename, input_names=input_names, output_names=output_names, opset_version=opset_version, *args, **kwargs)
    return Path(filename).exists()

import torch

from torch.onnx.symbolic_helper import parse_args, scalar_type_to_onnx, scalar_type_to_pytorch_type, \
    cast_pytorch_to_onnx
from torch.onnx.symbolic_opset9 import select, unsqueeze, squeeze, _cast_Long
from torch.onnx import register_custom_op_symbolic

from vortex.development.exporter.base_exporter import BaseExporter

class OnnxExporter(BaseExporter):
    def __init__(self, filename : str, image_size : int, input_dtype : str='uint8', n_channels=3, n_batch=1, opset_version=11, shape_inference=True, **kwargs) :
        super(OnnxExporter,self).__init__(
            image_size=image_size, input_dtype=input_dtype, 
            filename=filename, n_channels=n_channels, n_batch=n_batch,
        )
        self.export_args = kwargs
        self.opset_version = opset_version
        self.export_args.update({'opset_version' : opset_version})
        self.shape_inference = shape_inference
        """
        adapted from
        https://github.com/pytorch/vision/blob/74679cc566f98398db13df0312cc11188733f1f3/torchvision/ops/_register_onnx_ops.py#L7
        extend opset version
        """
        opset_version = [9,10]
        for _onnx_opset_version in opset_version :
            register_custom_op_symbolic('torchvision::nms', type(self).symbolic_multi_label_nms, _onnx_opset_version)
            register_custom_op_symbolic('torchvision::roi_align', type(self).roi_align, _onnx_opset_version)
            register_custom_op_symbolic('torchvision::roi_pool', type(self).roi_pool, _onnx_opset_version)
            ## broken on opset 10 :
            # register_custom_op_symbolic('torchvision::_new_empty_tensor_op', OnnxExporter.new_empty_tensor_op, _onnx_opset_version)

    @staticmethod
    @parse_args('v', 'v', 'f')
    def symbolic_multi_label_nms(g, boxes, scores, iou_threshold):
        boxes = unsqueeze(g, boxes, 0)
        scores = unsqueeze(g, unsqueeze(g, scores, 0), 0)
        max_output_per_class = g.op('Constant', value_t=torch.tensor([sys.maxsize], dtype=torch.long))
        iou_threshold = g.op('Constant', value_t=torch.tensor([iou_threshold], dtype=torch.float))
        nms_out = g.op('NonMaxSuppression', boxes, scores, max_output_per_class, iou_threshold)
        return squeeze(g, select(g, nms_out, 1, g.op('Constant', value_t=torch.tensor([2], dtype=torch.long))), 1)

    @staticmethod
    @parse_args('v', 'v', 'f', 'i', 'i', 'i', 'i')
    def roi_align(g, input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned):
        if(aligned):
            raise RuntimeError('Unsupported: ONNX export of roi_align with aligned')
        batch_indices = _cast_Long(g, squeeze(g, select(g, rois, 1, g.op('Constant',
                                   value_t=torch.tensor([0], dtype=torch.long))), 1), False)
        rois = select(g, rois, 1, g.op('Constant', value_t=torch.tensor([1, 2, 3, 4], dtype=torch.long)))
        return g.op('RoiAlign', input, rois, batch_indices, spatial_scale_f=spatial_scale,
                    output_height_i=pooled_height, output_width_i=pooled_width, sampling_ratio_i=sampling_ratio)

    @staticmethod
    @parse_args('v', 'v', 'f', 'i', 'i')
    def roi_pool(g, input, rois, spatial_scale, pooled_height, pooled_width):
        roi_pool = g.op('MaxRoiPool', input, rois,
                        pooled_shape_i=(pooled_height, pooled_width), spatial_scale_f=spatial_scale)
        return roi_pool, None

    @staticmethod
    @parse_args('v', 'is')
    def new_empty_tensor_op(g, input, shape):
        dtype = input.type().scalarType()
        if dtype is None:
            dtype = 'Float'
        dtype = scalar_type_to_onnx.index(cast_pytorch_to_onnx[dtype])
        shape = g.op("Constant", value_t=torch.tensor(shape))
        return g.op("ConstantOfShape", shape,
                    value_t=torch.tensor([0], dtype=scalar_type_to_pytorch_type[dtype]))
    
    def export(self, predictor, example_input, class_names, output_format, additional_inputs) :
        n_batch = self.image_size[0]
        postprocess = predictor.postprocess
        assert (n_batch > 1 and not isinstance(postprocess, BasicNMSPostProcess)) or (n_batch == 1) \
            or (isinstance(postprocess, BatchedNMSPostProcess) and n_batch > 1 and self.opset_version >= 11), \
            "batch input for detection model only supported on opset_version 11"
        self.output_name = ['output_{}'.format(i) for i in range(n_batch)] \
            if n_batch > 1 and isinstance(postprocess, BatchedNMSPostProcess) else ['output']
        if n_batch == 1 and isinstance(postprocess, BatchedNMSPostProcess):
            ## BatchedNMSPostProcess squeezed the output for each samples
            ## cast to super (BasicNMSPostProcess) to disable squeezing
            postprocess.__class__ = BasicNMSPostProcess

        input_names = ['input']
        inputs = [example_input]
        for additional_input in additional_inputs:
            input_name, shape = additional_input
            inputs.append(torch.tensor(0) if shape is None else torch.zeros(*shape))
            input_names.append(input_name)
        inputs = tuple(inputs)
        filename = str(self.filename)
        ok = export(
            model=predictor,
            example_input=inputs,
            filename=filename,
            input_names=input_names,
            output_names=self.output_name,
            **self.export_args
        )
        model = onnx.load(filename)
        get_op = GRAPH_OPS.create_from_args
        g_ops = [
            get_op('EmbedOutputFormat', output_format=output_format),
            get_op('EmbedClassNames', class_names=class_names)
        ]
        if isinstance(postprocess, (BatchedNMSPostProcess, BasicNMSPostProcess)) :
            g_ops.append(get_op('IOUThresholdAsInput'))
        if n_batch > 1 and isinstance(postprocess, BatchedNMSPostProcess) :
            g_ops.append(get_op('CreateBatchOutputSequence'))
        
        has_nms = len(get_Ops(model, 'NonMaxSuppression')[0]) > 0
        append_list = has_nms and n_batch > 1
        # doesnt support list for now
        if self.shape_inference and not append_list:
            g_ops.append(get_op('SymbolicShapeInfer'))

        for op in g_ops :
            model = op(model)

        onnx.save(model, filename)
        return ok
