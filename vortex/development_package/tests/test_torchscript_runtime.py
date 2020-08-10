import os
import cv2
import torch
import pytest
from pathlib import Path
from easydict import EasyDict

from vortex.development.exporter.torchscript import TorchScriptExporter
from vortex.development.networks.models import create_model_components
from vortex.development.predictor import create_predictor
from vortex.runtime.torchscript import TorchScriptRuntime
from vortex.runtime.torchscript import TorchScriptRuntimeCpu, TorchScriptRuntimeCuda
from collections import OrderedDict

project_dir = Path(__file__).parents[1]
output_dir = os.path.join(project_dir, "tmp", "torchscript")
os.makedirs(output_dir, exist_ok=True)

model_argmap = EasyDict(
    FPNSSD=dict(
        preprocess_args=dict(
            input_size=512,
            input_normalization=dict(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            )
        ),
        network_args=dict(
            backbone='resnet18',
            n_classes=20,
            pyramid_channels=256,
            aspect_ratios=[1, 2., 3.],
        ),
        loss_args=dict(
            neg_pos=3,
            overlap_thresh=0.5,
        ),
        postprocess_args=dict(
            nms=True,
        )
    ),
    RetinaFace=dict(
        preprocess_args=dict(
            input_size=640,
            input_normalization=dict(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            )
        ),
        network_args=dict(
            n_classes=1,
            backbone='resnet18',
            pyramid_channels=64,
            aspect_ratios=[1, 2., 3.],
        ),
        loss_args=dict(
            neg_pos=7,
            overlap_thresh=0.35,
            cls=2.0,
            box=1.0,
            ldm=1.0,
        ),
        postprocess_args=dict(
            nms=True,
        ),
    ),
    softmax=dict(
        network_args=dict(
            backbone='resnet18',
            n_classes=10,
            freeze_backbone=False,
        ),
        preprocess_args=dict(
            input_size=224,
            input_normalization=dict(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ),
        loss_args=dict(
            reduction='mean'
        ),
        postprocess_args={}
    )
)

def export_model(model_name):
    args = model_argmap[model_name]
    model = create_model_components(model_name, args['preprocess_args'], args['network_args'], 
        loss_args=args['loss_args'], postprocess_args=args['postprocess_args'])
    predictor = create_predictor(model)
    input_size = args['preprocess_args']['input_size']

    output_path = os.path.join(output_dir, '{}.pt'.format(model_name))
    exporter = TorchScriptExporter(output_path, image_size=input_size, check_tolerance=1e-6)
    result = exporter(predictor)
    assert os.path.exists(output_path)
    return output_path

@pytest.mark.parametrize(
    "model_name, device",
    [(name, device) for name in model_argmap for device in ("cpu", "cuda")]
)
def test_torchscript(model_name, device):
    assert TorchScriptRuntime.is_available(device), "Runtime for device '{}' is "\
        "not available".format(device)
    output_path = export_model(model_name)

    runtime = TorchScriptRuntime(output_path, device=device)
    n, h, w, c = runtime.input_specs["input"]["shape"]
    img = cv2.imread(os.path.join(project_dir, "tests", "images", "cat.jpg"))
    img = cv2.resize(img, (h, w))[None, :]

    kwargs = {"score_threshold": 0.05, "iou_threshold": 0.02}
    kwargs = {name: value for name, value in kwargs.items() if name in runtime.input_specs}
    result = runtime(img, **kwargs)
    assert isinstance(result, list)
    if len(result):
        assert isinstance(result[0], OrderedDict)

    os.remove(output_path)

def test_torchscript_failed_file():
    ## runtime should not accept model with extension other
    ## than '.pt' or '.pth'
    with pytest.raises(RuntimeError):
        filepath = project_dir.joinpath("tests", "test_torchscript_runtime.py")
        runtime = TorchScriptRuntime(filepath, device="cpu")
    with pytest.raises(RuntimeError):
        filepath = "experiments/configs/shufflenetv2x100_classification_cifar10.yml"
        runtime = TorchScriptRuntime(project_dir.joinpath(filepath), "cpu")

def test_torchscript_failed_module():
    ## runtime should not accept model of type `nn.Module`
    ## only accept `torch.jit.ScriptModule`
    model = torch.nn.Softmax(dim=1)
    with pytest.raises(RuntimeError):
        runtime = TorchScriptRuntime(model, device="cpu")
    with pytest.raises(RuntimeError):
        runtime = TorchScriptRuntime(model, device="cuda")

def test_torchscript_cpu():
    assert TorchScriptRuntimeCpu.is_available(), "Torchscript CPU runtime is not available"
    output_path = export_model("softmax")
    runtime = TorchScriptRuntimeCpu(output_path)
    os.remove(output_path)

def test_torchscript_cuda():
    assert TorchScriptRuntimeCuda.is_available(), "Torchscript CUDA runtime is not available"
    output_path = export_model("softmax")
    runtime = TorchScriptRuntimeCuda(output_path)
    os.remove(output_path)

def test_torchscript_cuda_invalid():
    device = torch.cuda.device_count()
    assert not TorchScriptRuntimeCuda.is_available(device)
    with pytest.raises(RuntimeError):
        output_path = export_model("softmax")
        runtime = TorchScriptRuntimeCuda(output_path, device_id=device)
        os.remove(output_path)
