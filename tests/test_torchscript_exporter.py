import os
import torch
import pytest
from easydict import EasyDict

from vortex.development.exporter.torchscript import TorchScriptExporter
from vortex.development.networks.models import create_model_components
from vortex.development.predictor import create_predictor


output_dir = "tmp/torchscript"
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


@pytest.mark.parametrize(
    "model_name, image",
    [(name, img) for name in model_argmap for img in (None, "tests/test_dataset/classification/val/cat/1.jpeg")]
)
def test_exporter(model_name, image, backbone="resnet18", remove_output=True):
    args = model_argmap[model_name]
    args.network_args.backbone = backbone
    model = create_model_components(model_name, args['preprocess_args'], args['network_args'], 
        loss_args=args['loss_args'], postprocess_args=args['postprocess_args'])
    predictor = create_predictor(model)
    input_size = args['preprocess_args']['input_size']
    output_path = os.path.join(output_dir, '{}_{}.pt'.format(model_name, backbone))
    output_format = predictor.output_format
    additional_inputs = None
    if hasattr(predictor.postprocess, 'additional_inputs'):
        additional_inputs = predictor.postprocess.additional_inputs

    print(" >> Exporting...")
    exporter = TorchScriptExporter(output_path, image_size=input_size, check_tolerance=1e-6)
    ok = exporter(predictor, example_image_path=image)
    assert ok and os.path.exists(output_path)
    del model, predictor, exporter

    model = torch.jit.load(output_path)
    for name, value in output_format.items():
        assert hasattr(model, name + '_axis')
        assert hasattr(model, name + '_indices')
        assert getattr(model, name + '_axis') == torch.tensor(value['axis'])
        assert all(getattr(model, name + '_indices') == torch.tensor(value['indices']))

    inputs = (('input', (1, input_size, input_size, 3)),)
    x = torch.randint(0, 256, (1,input_size,input_size,3))
    run_inputs, run_inputs_kwargs = [x], {}
    if additional_inputs:
        inputs += additional_inputs
    for n, (name, shape) in enumerate(inputs):
        assert hasattr(model, name + '_input_shape')
        assert hasattr(model, name + '_input_pos')
        assert getattr(model, name + '_input_shape').equal(torch.tensor(shape))
        assert getattr(model, name + '_input_pos').equal(torch.tensor(n))
        if name != 'input':
            run_inputs.append(torch.zeros(*shape))
            run_inputs_kwargs[name] = torch.zeros(*shape)
    shape_name = [name.replace('_input_shape', '') for name,_ in model.named_buffers(recurse=False) if name.endswith('_input_shape')]
    pos_name = [name.replace('_input_pos', '') for name,_ in model.named_buffers(recurse=False) if name.endswith('_input_pos')]
    assert sorted(shape_name) == sorted(pos_name)

    print(" >> Evaluating...")
    with torch.no_grad():
        out_eval = model(*run_inputs)
    assert len(out_eval) == 1 ## single batch
    assert out_eval[0].shape[-1] == sum(len(v['indices']) for v in output_format.values()) # number of elements

    ## TODO: check using keyword arguments, currently unsupported
    # out_eval_kwargs = model(x, **run_inputs_kwargs)
    # assert out_eval_kwargs.equal(out_eval)

    if remove_output:
        os.remove(output_path)
    del model, out_eval
