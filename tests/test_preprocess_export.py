import sys
sys.path.append('vortex/development_package')
sys.path.append('vortex/runtime_package')

import torch
import unittest
import numpy as np
import onnxruntime
from pathlib import Path


from vortex.runtime.onnx.onnxruntime import OnnxRuntimeCpu as Runtime
from vortex.development.exporter.onnx import export
from vortex.development.networks.modules.preprocess import get_preprocess

class PreprocessTest(unittest.TestCase) :
    def test_normalize_torchscript(self) :
        mean, std,scaler =[0.485, 0.456, 0.406], [0.229, 0.224, 0.225],255
        normalizer = get_preprocess('normalizer',mean=mean,std=std,scaler=scaler)
        normalizer_script = torch.jit.script(normalizer)
        x = (torch.rand(480,640,3) * 255).type(torch.uint8).unsqueeze(0)
        x = normalizer_script(x)
        self.assertEqual(x.size(),torch.Size([1,3,480,640]))
        min_value = [(0. - mean[i]) / std[i] for i in range(len(mean))]
        max_value = [(1. - mean[i]) / std[i] for i in range(len(mean))]
        self.assertEqual(x.size(),torch.Size([1,3,480,640]))
        self.assertTrue(all(
            [torch.all(x[:,i,:,:] >= min_value[i]) and torch.all(x[:,i,:,:] <= max_value[i]) for i in range(len(mean))]
        ))
    def test_normalize_onnx(self) :
        mean, std,scaler =[0.485, 0.456, 0.406], [0.229, 0.224, 0.225],255
        normalizer = get_preprocess('normalizer',mean=mean,std=std,scaler=scaler)
        filename = 'test_normalize.onnx'
        x = (torch.rand(480,640,3) * 255).type(torch.uint8).unsqueeze(0)
        self.assertTrue(export(normalizer,x,filename,input_names=['input'],output_names=['normalized']))
        session = onnxruntime.InferenceSession(filename)
        ## TODO : provide more generic runtime
        torch_output = normalizer(x)
        self.assertEqual(torch_output.size(),torch.Size([1,3,480,640]))
        # onnx_output = session(None, {'input' : x.numpy().astype(np.uint8)})
        onnx_output = session.run(['normalized'],{'input' : x.numpy().astype(np.uint8)})
        onnx_output = onnx_output[0]
        self.assertEqual(onnx_output.shape,(1,3,480,640))
        self.assertTrue(
            np.allclose(torch_output.detach().numpy(),onnx_output)
        )
        Path(filename).unlink()