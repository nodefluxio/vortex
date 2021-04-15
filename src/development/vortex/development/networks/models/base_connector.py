import torch.nn as nn

from vortex.development.networks.modules.backbones import get_backbone
from vortex.development.networks.modules.backbones.base_backbone import supported_feature_type

class BackbonePoolConnector(nn.Module):
    def __init__(self, backbone: str, feature_type='tri_stage_fpn',
                 pretrained_backbone=False, freeze_backbone=False, *args, **kwargs):
        super(BackbonePoolConnector, self).__init__()
        if not isinstance(backbone, str):
            raise RuntimeError(
                "expects backbone to be 'str' instance, got %s" % (type(backbone)))
        if not feature_type in supported_feature_type:
            raise RuntimeError("feature type of '%s' is not supported, "\
                "supported: %s" % (feature_type, supported_feature_type))

        if isinstance(backbone, str):
            if feature_type == "classifier" and "n_classes" not in kwargs:
                ## number of classes in backbone only required for classifier feature type
                raise RuntimeError("'n_classes' argument in model for classifier feature "\
                    "is not defined, make sure you have define it properly.")
            backbone = get_backbone(backbone, pretrained=pretrained_backbone, 
                feature_type=feature_type, *args, **kwargs)
        self.backbone = backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)
