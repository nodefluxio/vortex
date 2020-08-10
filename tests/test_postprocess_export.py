import sys
sys.path.append('vortex/development_package')

import torch
import unittest
import numpy as np
import onnxruntime
from pathlib import Path

from vortex.runtime.onnx.onnxruntime import OnnxRuntimeCpu as Runtime
from vortex.development.exporter.onnx import export
from vortex.development.networks.modules.postprocess import get_postprocess

class PostprocessTest(unittest.TestCase) :
    def test_yolo_postprocess_onnx(self) :
        postprocess = get_postprocess('yolov3')
        filename = 'test_postprocess_yolo.onnx'
        x = torch.rand(1,1000,9)
        example_input = (x.clone(), torch.tensor([0.5]), torch.tensor([0.5]))
        dynamic_axes = {'output' : {1 : 'detections'}}
        opset_version = 11
        input_names = ['input','score_threshold','iou_threshold']
        output_names = ['output']
        self.assertTrue(export(
            postprocess,
            example_input,
            filename,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes
        ))
        session = onnxruntime.InferenceSession(filename)
        onnx_output = session.run(['output'], {k : t.numpy() for k, t in zip(input_names,example_input)})
        torch_output = postprocess(*example_input)
        self.assertEqual(len(onnx_output[0].shape), 3)
        self.assertTrue(
            np.allclose(torch_output[0].numpy(),onnx_output[0])
        )
        import pathlib
        pathlib.Path(filename).unlink()