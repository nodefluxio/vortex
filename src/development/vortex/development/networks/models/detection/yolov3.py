import torch

from ..base_connector import BackbonePoolConnector
from ...modules.backbones.base_backbone import Backbone
from ...modules.heads.detection.yolov3 import YoloV3Head
from ...modules.utils.darknet import yolo_feature_maps

from easydict import EasyDict
from typing import List, Tuple, Union, Type


__all__ = [
    'YoloV3'
]

supported_models = [
    'YoloV3'
]


class YoloV3(BackbonePoolConnector):
    """
    x -> backbone -> (c3, c4, c5) -> head -> (h1, h2, h3)
    """

    coco_anchors = [
        (10, 13),  (16, 30),  (33, 23),  (30, 61),  (62, 45),
        (59, 119),  (116, 90),  (156, 198),  (373, 326)
    ]

    def __init__(self, backbone: Union[str, Type[Backbone]], img_size: int, n_classes: int, 
                 anchors: Union[List[Tuple[int, int]]] = None, *args, **kwargs):
        super(YoloV3, self).__init__(backbone, feature_type="tri_stage_fpn", *args, **kwargs)
        downsampling = [2**n for n in range(5)]
        grids = yolo_feature_maps(img_size)
        backbone_channels = [int(x) for x in self.backbone.out_channels]
        if not anchors:
            anchors = self.coco_anchors
        self.head = YoloV3Head(
            img_size=img_size,
            backbone_channels=backbone_channels,
            grids=grids,
            anchors=anchors,
            n_classes=n_classes
        )
        self.grids = grids
        self.task = "detection"
        self.output_format = {
            "bounding_box": {"indices": [0,1,2,3], "axis": 1},
            "class_confidence": {"indices": [4], "axis": 1},
            "class_label": {"indices": [5], "axis": 1},
        }

    def get_anchors(self):
        return self.head.get_anchors()

    def forward(self, x: torch.Tensor) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        s3, s4, s5 = self.backbone(x)
        h1, h2, h3 = self.head(s3, s4, s5)
        if self.training:
            output = h1, h2, h3
        else:
            output = torch.cat((h1, h2, h3), 1)
        return output

    @staticmethod
    def postprocess(preds: Tuple[torch.Tensor]):
        """
        postprocess output from forward into N, features, N_class + 5
        example, with class 80
        torch.Size([8, 10647, 85])
        Args:
            preds:

        Returns:

        """
        out = torch.cat((preds[0], preds[1], preds[2]), 1)
        return out


def create_model_components(preprocess_args: EasyDict, network_args: EasyDict, 
        loss_args: EasyDict, postprocess_args: EasyDict) -> EasyDict:

    from vortex.development.networks.modules.losses.yolov3 import YoloV3Loss, YoloV3LossDena
    from vortex.development.networks.modules.postprocess.yolov3 import YoloV3PostProcess
    img_size = preprocess_args.input_size
    model_components = EasyDict()
    model_components.network = YoloV3(img_size=img_size, **network_args)
    # model_components.loss = YoloV3Loss(img_size=img_size, **loss_args)
    # model_components.loss.assign_anchors(model_components.network.get_anchors())
    # model_components.collate_fn = 'DarknetCollate'
    model_components.loss = YoloV3LossDena(img_size, 
        anchors=model_components.network.get_anchors(), 
        ref_anchors=model_components.network.head.get_ref_anchors(), 
        mask_anchors=model_components.network.head.get_anchor_masks(),
        **loss_args)
    model_components.collate_fn = 'DarknetCollateNewDim'
    model_components.postprocess = YoloV3PostProcess(img_size=img_size, **postprocess_args)
    return model_components
