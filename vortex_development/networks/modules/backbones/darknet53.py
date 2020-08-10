import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import enforce

from typing import Tuple, List, Union
from .base_backbone import Backbone, ClassifierFeature
from vortex.networks.modules.utils.darknet import DarknetResidual, darknet_conv

from pathlib import Path

__all__ = [
    'darknet53',
    'supported_models',
    'get_model',
]

supported_models = [
    'darknet53'
]


class DarknetClassifier(nn.Module):
    def __init__(self, n_classes):
        super(DarknetClassifier, self).__init__()
        self.classifier = nn.Linear(1024, n_classes)
    
    def forward(self, x):
        x = x.mean([2, 3])  # global pool (see torchvision shufflenetv2)
        x = self.classifier(x)
        return x

class Darknet53(nn.Module):
    """
    # Stem
        [convolutional]
        batch_normalize=1
        filters=32
        size=3
        stride=1
        pad=1
        activation=leaky
    # Downsample
        [convolutional]
        batch_normalize=1
        filters=int
        size=3
        stride=2
        pad=1
        activation=leaky
    # DarknetResidual
        [convolutional]
        batch_normalize=1
        filters=int
        size=1
        stride=1
        pad=1
        activation=leaky
        [convolutional]
        batch_normalize=1
        filters=int
        size=3
        stride=1
        pad=1
        activation=leaky
        [shortcut]
        from=-3
        activation=linear
    """
    @staticmethod
    def residual_block(repeat: int, *args, **kwargs):
        layers = []
        for i in range(0, repeat):
            layers.append(DarknetResidual(*args, **kwargs))
        return nn.Sequential(*layers)

    def __init__(self, n_classes: int = 1000):
        super(Darknet53, self).__init__()
        downsampling_kwargs = {
            'bn': True,
            'kernel_size': 3,
            'stride': 2,
            'pad': True,
            'activation': 'leaky',
        }
        self.blocks = nn.Sequential(
            nn.Sequential(              # 1st spatial downsampling
                darknet_conv(           # stem
                    3, 32, bn=True,
                    kernel_size=3, pad=True, stride=1,
                    activation='leaky'
                ),
                darknet_conv(32, 64, **downsampling_kwargs),
                Darknet53.residual_block(
                    repeat=1, in_channels=64, filters=(32, 64)
                ),
            ),
            nn.Sequential(              # 2nd spatial downsampling
                darknet_conv(64, 128, **downsampling_kwargs),
                Darknet53.residual_block(
                    repeat=2, in_channels=128, filters=(64, 128)
                ),
            ),
            nn.Sequential(              # 3rd spatial downsampling
                darknet_conv(128, 256, **downsampling_kwargs),
                Darknet53.residual_block(
                    repeat=8, in_channels=256, filters=(128, 256)
                ),
            ),
            nn.Sequential(              # 4th spatial downsampling
                darknet_conv(256, 512, **downsampling_kwargs),
                Darknet53.residual_block(
                    repeat=8, in_channels=512, filters=(256, 512)
                ),
            ),
            nn.Sequential(              # 5th spatial downsampling
                darknet_conv(512, 1024, **downsampling_kwargs),
                Darknet53.residual_block(
                    repeat=4, in_channels=1024, filters=(512, 1024)
                ),
            ),
        )
        self.classifier = DarknetClassifier(n_classes)

    def forward(self, x: torch.Tensor):
        x = self.blocks(x)
        x = self.classifier(x)
        return F.softmax(x, dim=0)


def darknet53(*args, **kwargs):
    return Darknet53(*args, **kwargs)


@enforce.runtime_validation
def get_model(model: str, pretrained: Union[str, Path, None, bool] = None, *args, **kwargs):
    assert(model in supported_models), "%s not supported! %s" % (
        model, supported_models)
    # TODO : consider using eval (?)
    if model == 'darknet53':
        model = darknet53(*args, **kwargs)
    if pretrained:
        pretrained_file = Path(pretrained)
        assert(pretrained_file.exists())
        model.load_state_dict(torch.load(str(pretrained_file)))
    return model


@enforce.runtime_validation
def get_backbone(model_name: str, pretrained: Union[str, Path, None, bool] = None, 
                 feature_type: str = "tri_stage_fpn", n_classes: int = 1000,
                 *args, **kwargs):
    model = get_model(model_name, pretrained=pretrained, n_classes=n_classes, *args, *kwargs)
    n_channels = [64, 128, 256, 512, 1024]

    if feature_type == "tri_stage_fpn":
        backbone = Backbone(model.blocks, n_channels)
    elif feature_type == "classifier":
        backbone = ClassifierFeature(model.blocks, model.classifier, n_classes)
    else:
        raise NotImplementedError("'feature_type' for other than 'tri_stage_fpn' and 'classifier'"\
            "is not currently implemented, got %s" % (feature_type))
    return backbone
