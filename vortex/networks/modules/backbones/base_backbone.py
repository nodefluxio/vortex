import torch
import torch.nn as nn

from typing import Tuple, Sequence, Type

supported_feature_type = ['tri_stage_fpn', 'classifier']

class Backbone(nn.Module):
    """
    backbone adapter with channels information
    for tri_stage_fpn feature
    """
    n_stage = 5
    n_output = 3

    def __init__(self, stages: nn.Sequential, channels: Sequence[int]):
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
    def __init__(self, stages: nn.Sequential, classifier: Type[nn.Module], channels: int):
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
