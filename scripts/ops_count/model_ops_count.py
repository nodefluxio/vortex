import os, sys, inspect
from pathlib import Path
import torch
import pandas as pd

repo_root = Path(__file__).parent.parent

if __name__=='__main__' :
    sys.path.insert(0, str(repo_root))

from vortex.networks.modules import backbones
from vortex.networks import models
from thop import profile
from mod_torchsummary import summary
from easydict import EasyDict
from vortex.networks.models import create_model_components

mean_std=dict(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],
)
model_argmap = EasyDict(
    FPNSSD=dict(
        preprocess_args=dict(
            input_size=448,
            input_normalization=mean_std
        ),
        network_args=dict(
            backbone='darknet53',
            n_classes=1,
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
            input_size=448,
            input_normalization=mean_std
        ),
        network_args=dict(
            n_classes=1,
            backbone='darknet53',
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
    )
)

def create_model_cfg(model_name, model_arg) :
    model_arg = EasyDict(model_arg[model_name])
    config = {
        'name' : model_name,
        **model_arg
    }
    return EasyDict(config)

def calculate_backbone_params_and_flops():
    all_supported_backbones=backbones.all_models
    backbone_df = pd.DataFrame(columns=['backbone_name', 'params', 'mac/flops'])
    for backbone in all_supported_backbones:
        network = backbones.get_backbone(backbone, pretrained=False, feature_type='tri_stage_fpn', n_classes=1000)
        input = torch.randn(1, 3, 224, 224)
        macs, _ = profile(network, inputs=(input, ),verbose=False)
        params = summary(network,(3,224,224),device='cpu')[0].item()
        macs_in_g=macs/(10**9)
        params_in_m=params/(10**6)
        print(backbone,params_in_m,macs_in_g)
        backbone_df = backbone_df.append({'backbone_name': backbone,
                                          'params': params_in_m,
                                          'mac/flops': macs_in_g}, ignore_index=True)
    backbone_df.to_csv('backbones_summary.txt',index=False)

def calculate_models_params_and_flops():

    backbone='darknet53'
    network = backbones.get_backbone(backbone, pretrained=False, feature_type='tri_stage_fpn', n_classes=1000)
    input = torch.randn(1, 3, 448, 448)
    bb_macs, _ = profile(network, inputs=(input, ),verbose=False)
    bb_params = summary(network,(3,448,448),device='cpu')[0].item()
    model_df = pd.DataFrame(columns=['model_name', 'params', 'mac/flops'])
    for key in model_argmap:
        model_cfg= create_model_cfg(key, model_argmap)
        model_components = create_model_components(model_cfg['name'],
                                                   preprocess_args=model_cfg['preprocess_args'],
                                                   network_args=model_cfg['network_args'],
                                                   loss_args=model_cfg['loss_args'],
                                                   postprocess_args=model_cfg['postprocess_args'])
        network=model_components.network
        model_macs, model_params = profile(network, inputs=(input, ),verbose=False)
        if 'backbone' in model_cfg.network_args:
            model_macs-=bb_macs
            model_params-=bb_params
            key+='_head'
        model_macs_in_g=model_macs/(10**9)
        model_params_in_m=model_params/(10**6)
        print(key,model_params_in_m,model_macs_in_g)
        model_df = model_df.append({'model_name': key,
                                    'params': model_params_in_m,
                                    'mac/flops': model_macs_in_g}, ignore_index=True)
    model_df.to_csv('models_summary.txt',index=False)
if __name__ == "__main__":
    calculate_backbone_params_and_flops()
    calculate_models_params_and_flops()
