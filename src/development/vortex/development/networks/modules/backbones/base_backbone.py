import torch
import torch.nn as nn

from typing import Tuple, Sequence, Type, Union, NamedTuple
from abc import ABC, abstractmethod

from ..utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class BackboneConfig(NamedTuple):
    pretrained_url: str = None
    normalize_mean: tuple = IMAGENET_DEFAULT_MEAN
    normalize_std: tuple = IMAGENET_DEFAULT_STD
    input_size: tuple = (3, 224, 224)
    channel_order: str = 'rgb'  # 'rgb', 'bgr'
    resize: str = 'stretch'     # 'stretch', 'pad', 'scale'
    num_classes: int = 1000


class BackboneBase(ABC, nn.Module):
    def __init__(self, name: str, default_config: BackboneConfig = None):
        super().__init__()

        if default_config is None:
            default_config = BackboneConfig()
        elif not isinstance(default_config, BackboneConfig):
            raise TypeError("'default_config' argument is expected to have 'BackboneConfig' type, "
                "got {}.".format(type(default_config)))
        self._default_cfg = default_config

        if not isinstance(name, str):
            raise RuntimeError("'name' argument is expected to have 'str' type, got {}".format(type(name)))
        self._name = name

        self._inferred_channels = None

    @abstractmethod
    def get_stages(self) -> Union[nn.Sequential, nn.ModuleList]:
        pass

    @property
    def stages_channel(self) -> Tuple:
        if self._inferred_channels is None:
            self._inferred_channels = infer_channels(self.get_stages())
        return self._inferred_channels

    @abstractmethod
    def get_classifier(self) -> nn.Module:
        pass

    @abstractmethod
    def reset_classifier(self, num_classes: int, classifier: nn.Module = None):
        """Resets the classifier layer
        normally used to change number of classes in transfer learning.

        Args:
            num_classes (int): number of classes for the classifier to be reset to.
            classifier (Union[int, nn.Module]): override classifier layer with this layer.
        """
        pass

    @property
    def default_config(self) -> BackboneConfig:
        return self._default_cfg

    @default_config.setter
    def default_config(self, val):
        self._default_cfg = val

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        self._name = val

    @property
    @abstractmethod
    def num_classes(self) -> int:
        pass

    @property
    @abstractmethod
    def num_classifer_feature(self) -> int:
        pass


supported_feature_type = ['tri_stage_fpn', 'classifier']

class Backbone(nn.Module):
    """
    backbone adapter with channels information
    for tri_stage_fpn feature
    """
    n_stage = 5
    n_output = 3

    def __init__(self, stages: nn.Sequential, channels: Union[Sequence[int]]=None):
        if channels is None:
            channels = infer_channels(stages)
        if not len(stages) == len(channels):
            raise RuntimeError(
                "expects len(stages) == len(channels) got %s and %s" % (len(stages), len(channels)))
        if not len(channels) == Backbone.n_stage:
            raise RuntimeError("expects n stage == %s got %s" %
                               (Backbone.n_stage, len(channels)))
        super(Backbone, self).__init__()
        stage1, stage2, stage3, stage4, stage5 = stages
        self.stage1 = stage1
        self.stage2 = stage2
        self.stage3 = stage3
        self.stage4 = stage4
        self.stage5 = stage5

        self.out_channels = tuple(channels)
        self.feature_type = 'tri_stage_fpn'

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)
        return (s3, s4, s5)


class ClassifierFeature(nn.Module):
    """
    backbone adapter with channels information
    for classifier feature
    """
    def __init__(self, stages: nn.Sequential, classifier: Type[nn.Module], channels: Union[int]=None):
        if channels is None:
            ## get last channel
            channels = infer_channels(stages)[-1]
        if not isinstance(channels, int):
            raise RuntimeError("expects `channels` for `ClassifierFeature` to have int type, "\
                "got %s" % type(channels))

        super().__init__()
        self.stages = stages
        self.classifier = classifier

        self.out_channels = channels
        self.feature_type = 'classifier'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stages(x)
        x = self.classifier(x)
        return x

def infer_channels(stages: nn.Sequential, test_size=(1,3,224,224)):
    """
    Helper function to infer the number of output channels from sequential module
    Args:
        stages: torch sequential module
        test_size: shape for initial input shape
    Return:
        channels: list of int, containing inferred output conv channels
    """
    channels = []
    x = torch.rand(test_size)
    with torch.no_grad():
        for stage in stages:
            x = stage(x)
            channels.append(x.shape)
    return tuple(map(lambda x: x[1], channels))
