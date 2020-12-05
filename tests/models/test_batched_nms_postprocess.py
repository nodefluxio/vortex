import torch
import pytest
from pathlib import Path

from vortex.development.networks.models.detection.retinaface import DefaultBox
from vortex.development.networks.modules.postprocess.retinaface import RetinaFacePostProcess as PostProcess

class Wrapper(torch.nn.Module) :
    def __init__(self,pp) :
        super(type(self),self).__init__()
        self.pp = pp
    
    def forward(self, input, score_threshold, iou_threshold) :
        return self.pp(input,score_threshold,iou_threshold)

@pytest.mark.skip(reason='cast to tuple brokes scripting')
def test_retinaface_postprocess_script() :
    anchor_gen =  DefaultBox(
        image_size=640,
        aspect_ratios=[1],
        variance=[0.1,0.2],
        steps=[8,16,32],
        anchors=None
    )
    pp = PostProcess(
        priors=anchor_gen(), 
        variance=torch.tensor([0.1,0.2]), 
        n_landmarks=5,
    )
    scripted = torch.jit.script(pp)

def test_retinaface_postprocess_trace() :
    anchor_gen =  DefaultBox(
        image_size=640,
        aspect_ratios=[1],
        variance=[0.1,0.2],
        steps=[8,16,32],
        anchors=None
    )
    pp = PostProcess(
        priors=anchor_gen(), 
        variance=torch.tensor([0.1,0.2]), 
        n_landmarks=5,
    )
    traced = torch.jit.trace(pp, example_inputs=(
        torch.rand(1,16800,16), torch.tensor([0.6]), torch.tensor([0.2])
    ))

@pytest.mark.skip(reason='cast to tuple brokes scripting')
def test_retinaface_postprocess_wrapped_script() :
    anchor_gen =  DefaultBox(
        image_size=640,
        aspect_ratios=[1],
        variance=[0.1,0.2],
        steps=[8,16,32],
        anchors=None
    )
    pp = PostProcess(
        priors=anchor_gen(), 
        variance=torch.tensor([0.1,0.2]), 
        n_landmarks=5,
    )
    wrapper = Wrapper(pp)
    wrapper.pp = torch.jit.script(wrapper.pp)
    traced = torch.jit.script(wrapper)

@pytest.mark.skip(reason='cast to tuple brokes scripting')
def test_retinaface_postprocess_wrapped_trace() :
    anchor_gen =  DefaultBox(
        image_size=640,
        aspect_ratios=[1],
        variance=[0.1,0.2],
        steps=[8,16,32],
        anchors=None
    )
    pp = PostProcess(
        priors=anchor_gen(), 
        variance=torch.tensor([0.1,0.2]), 
        n_landmarks=5,
    )
    wrapper = Wrapper(pp)
    wrapper.pp = torch.jit.script(wrapper.pp)
    traced = torch.jit.trace(pp, example_inputs=(
        torch.rand(1,16800,16), torch.tensor([0.6]), torch.tensor([0.2])
    ))

def test_retinaface_postprocess_onnx() :
    anchor_gen =  DefaultBox(
        image_size=640,
        aspect_ratios=[1],
        variance=[0.1,0.2],
        steps=[8,16,32],
        anchors=None
    )
    pp = PostProcess(
        priors=anchor_gen(), 
        variance=torch.tensor([0.1,0.2]), 
        n_landmarks=5,
    )
    traced = torch.onnx.export(pp, args=(
        torch.rand(1,16800,16), torch.tensor([0.6]), torch.tensor([0.2])
    ), f='test.onnx', opset_version=11)
    Path('test.onnx').unlink()