import torch
import cv2
import numpy as np
import os

from typing import Union
from vortex.networks.modules.postprocess.utils import nms
from vortex.networks.modules.postprocess.base_postprocess import BasicNMSPostProcess, BatchedNMSPostProcess

from vortex.exporter.base_exporter import BaseExporter

class TorchScriptExporter(BaseExporter):

    def __init__(self, filename: str, image_size: int, input_dtype: str = 'uint8', 
                 n_channels=3, n_batch=1, check_tolerance:Union[float,str]=1e-6, **kwargs):
        if not isinstance(filename, str):
            filename = str(filename)
        if len(filename.split('.')) == 1:
            filename += '.pt'
        super(TorchScriptExporter,self).__init__(n_batch=n_batch,
            filename=filename, input_dtype=input_dtype, 
            image_size=image_size, n_channels=n_channels, 
        )
        self.export_args = kwargs
        self.export_args.update({'check_tolerance' : float(check_tolerance)})

    def export(self, predictor, example_input, class_names, output_format, additional_inputs) :
        predictor = predictor.eval()

        inputs = [example_input]
        # input_spec -> (shape, position)
        input_spec = {'input': (torch.tensor(self.image_size), torch.tensor(0))}
        for n, (name, shape) in enumerate(additional_inputs):
            inputs.append(torch.zeros(*shape))
            input_spec[name] = (torch.tensor(shape), torch.tensor(n+1))

        ## this is a crude workaround for zero detection error in nms
        ## TODO: find a better solution
        if isinstance(predictor.postprocess, BasicNMSPostProcess) and \
                not isinstance(predictor.postprocess.nms, nms.NoNMS):
            predictor.postprocess.nms.nms_fn = torch.jit.script(predictor.postprocess.nms.nms_fn)

        ## TODO: support for keyword argument as input
        type(self).embed_input_spec(predictor, input_spec)
        type(self).embed_output_format(predictor, output_format)
        type(self).embed_class_names(predictor, class_names)
        exported = torch.jit.trace(predictor, example_inputs=tuple(inputs), 
            **self.export_args)
        exported.save(self.filename)
        return os.path.exists(self.filename)

    @staticmethod
    def embed_output_format(predictor, output_format):
        for name, value in output_format.items():
            assert 'indices' and 'axis' in value
            indices = torch.tensor(value['indices'])
            axis = torch.tensor(value['axis'])
            indices_name = name + '_indices'
            axis_name = name + '_axis'
            predictor.register_buffer(indices_name, indices)
            predictor.register_buffer(axis_name, axis)

    @staticmethod
    def embed_input_spec(predictor, input_spec):
        assert isinstance(list(input_spec.values())[0], tuple)
        for name, (shape, pos) in input_spec.items():
            predictor.register_buffer(name + '_input_shape', shape)
            predictor.register_buffer(name + '_input_pos', pos)

    @staticmethod
    def embed_class_names(predictor, class_names : dict):
        assert isinstance(class_names, dict)
        for key, value in class_names.items() :
            predictor.register_buffer('{}_label'.format(key), torch.as_tensor(value))