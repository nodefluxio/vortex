import torch
import torch.nn as nn

from typing import Union, Tuple, List

from ...utils.darknet import darknet_conv

__all__ = [
    'YoloV3Head',
    'YoloV3Layer',
    'YoloV3LayerConv',
    'YoloV3ConvBlock',
    'YoloV3UpsampleBlock',
]


supported_models = [
    'YoloV3Head'
]


class YoloV3ConvBlock(nn.Module):
    @staticmethod
    def get_yolo_conv_layers(in_channels: int, out_channels: int):
        """
        ######################

        [convolutional]
        batch_normalize=1
        filters=512
        size=1
        stride=1
        pad=1
        activation=leaky

        [convolutional]
        batch_normalize=1
        size=3
        stride=1
        pad=1
        filters=1024
        activation=leaky

        repeat 2x

        [convolutional]
        batch_normalize=1
        filters=512
        size=1
        stride=1
        pad=1
        activation=leaky
        """
        return nn.Sequential(
            darknet_conv(
                in_channels=in_channels,
                filters=out_channels//2, kernel_size=1,
                stride=1, pad=True,
                activation='leaky', bn=True
            ),
            darknet_conv(
                in_channels=out_channels//2,
                filters=out_channels, kernel_size=3,
                stride=1, pad=True,
                activation='leaky', bn=True
            ),
            darknet_conv(
                in_channels=out_channels,
                filters=out_channels//2, kernel_size=1,
                stride=1, pad=True,
                activation='leaky', bn=True
            ),
                darknet_conv(
                in_channels=out_channels//2,
                filters=out_channels, kernel_size=3,
                stride=1, pad=True,
                activation='leaky', bn=True
            ),
            darknet_conv(
                in_channels=out_channels,
                filters=out_channels//2, kernel_size=1,
                stride=1, pad=True,
                activation='leaky', bn=True
            )
        )

    def __init__(self, in_channels: int, out_channels: int):
        super(YoloV3ConvBlock, self).__init__()
        self.conv = YoloV3ConvBlock.get_yolo_conv_layers(in_channels, out_channels)

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class YoloV3UpsampleBlock(nn.Module):
    """
    yolo upsample block (upsample(x) + concat)
    x -> conv -> upsample --- |
            |                 | cat -> out2 (to next upsample)
            |      route ---- |
            |
            out1 (to detection)
    expected shape :
        x : [N,C,H,W]
        rout : [N,Cr,H*2,W*2]
        out1 : [N,C//2,H*2,W*2]
        out2 : [N,C//4+Cr,H*2,W*2]
    """
    @staticmethod
    def get_upsample_layer(in_channels: int, scale: int = 2):
        """
        [route]
        layers = -4

        [convolutional]
        batch_normalize=1
        filters=256
        size=1
        stride=1
        pad=1
        activation=leaky

        [upsample]
        stride=2

        [route]
        layers = -1, 61
        """
        return nn.Sequential(
            darknet_conv(
                in_channels=in_channels,
                filters=in_channels//2, kernel_size=1,
                stride=1, pad=True,
                activation='leaky', bn=True
            ),
            nn.Upsample(
                scale_factor=scale,
                mode='nearest'
            )
        )

    def __init__(self, in_channels: int, out_channels: int, scale: int = 2):
        super(YoloV3UpsampleBlock, self).__init__()
        self.conv = YoloV3ConvBlock(in_channels=in_channels, out_channels=out_channels)
        self.upsample = YoloV3UpsampleBlock.get_upsample_layer(out_channels//2, scale)

    def forward(self, x: torch.Tensor, route: torch.Tensor):
        out1 = self.conv(x)
        out2 = torch.cat((self.upsample(out1), route), 1)
        return out1, out2


class YoloV3LayerConv(nn.Module):
    """
    [convolutional]
    batch_normalize=1
    size=3
    stride=1
    pad=1
    filters=1024
    activation=leaky

    [convolutional]
    size=1
    stride=1
    pad=1
    filters=18
    activation=linear
    """
    @staticmethod
    def get_head_conv(in_channels: int, n_classes: int):
        return nn.Sequential(
            darknet_conv(
                in_channels=in_channels,
                filters=in_channels*2,
                kernel_size=3, stride=1,
                pad=True, activation='leaky', bn=True
            ),
            nn.Conv2d(in_channels=in_channels*2,
                              out_channels=3*(n_classes + 5),
                              kernel_size=1, stride=1, padding=0)
        )

    def __init__(self, in_channels: int, n_classes: int):
        super(YoloV3LayerConv, self).__init__()
        self.conv = YoloV3LayerConv.get_head_conv(in_channels, n_classes)

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class YoloV3Layer(nn.Module):
    """
    [yolo]
    mask = 6,7,8
    anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
    classes=1
    """
    __constants__ = ['grid_xy', 'stride', 'anchor_wh',
                     'anchor_vec', 'anchors', 'ref_anchors']

    def __init__(self, img_size: Union[int, List[int]], grids: Tuple[int, int], mask: List[int], anchors: List[Tuple[int, int]], n_classes: int):
        super(YoloV3Layer, self).__init__()

        if type(img_size) == list:
            img_size = max(img_size)
        stride = img_size // grids[0]
        self.all_anchors_grid = [
            (w/stride, h/stride) for w, h in anchors
        ]
        ref_anchors = torch.zeros(len(self.all_anchors_grid), 4)
        ref_anchors[:, 2:] = torch.tensor(self.all_anchors_grid)
        self.register_buffer('ref_anchors', ref_anchors)
        self.anchor_mask = mask

        anchors = torch.Tensor([anchors[i] for i in mask])
        self.register_buffer('anchors', anchors)
        self.n_anchors = len(mask)  # number of anchors (3)
        self.n_classes = n_classes  # number of classes (80)
        ny, nx, grid_xy, stride, anchor_wh = self.create_grids(
            anchors, img_size, grids, len(anchors)
        )
        self.ny, self.nx = ny, nx
        self.register_buffer('anchor_wh', anchor_wh)
        self.register_buffer('grid_xy', grid_xy)
        self.register_buffer('stride', stride)
        # scaled anchors
        anchor_vec = self.anchors / self.stride
        self.register_buffer('anchor_vec', anchor_vec)
        self.n_predict = self.n_classes + 5

    @staticmethod
    def create_grids(anchors, img_size, n_grids, n_anchors, dtype: torch.dtype = torch.float32):
        nx, ny = n_grids  # x and y grid size
        img_size = img_size
        stride = torch.tensor([img_size / max(n_grids)])

        # build xy offsets
        yv, xv = torch.meshgrid(
            [torch.arange(ny).type(dtype), torch.arange(nx).type(dtype)])
        grid_xy = torch.stack((xv, yv), 2).type(dtype).view((1, 1, ny, nx, 2))

        # build wh gains
        anchor_vec = anchors / stride
        anchor_wh = anchor_vec.view(1, n_anchors, 1, 1, 2).type(dtype)

        return nx, ny, grid_xy, stride, anchor_wh

    @staticmethod
    @torch.jit.script
    def recompute_grids(n_grids: torch.Tensor):
        assert(len(n_grids) == 2)
        # build xy offsets
        nx, ny = n_grids[0], n_grids[1]
        yv, xv = torch.meshgrid([
            torch.arange(ny, dtype=torch.float), 
            torch.arange(nx, dtype=torch.float)
        ])
        grid_xy = torch.stack((xv, yv), 2).view(
            (1, 1, int(ny), int(nx), 2)).to(n_grids.device)
        return grid_xy

    def forward(self, prediction: torch.Tensor):
        assert(len(prediction.shape) == 4)
        batch_size, ch, nx, ny = prediction.shape

        if self.training and nx != self.nx:
            grid_xy = self.recompute_grids(
                torch.tensor([nx, ny], device=prediction.device)
            )
            self.nx, self.ny = nx, ny
            self.grid_xy = grid_xy

        prediction = prediction.view(
            batch_size,
            self.n_anchors,
            self.n_predict,
            self.ny, self.nx
        ).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        xy = torch.sigmoid(prediction[..., 0:2])
        wh = prediction[..., 2:4]
        pred_conf = torch.sigmoid(prediction[..., 4]) \
            .view(batch_size, self.n_anchors, self.nx, self.ny, 1)
        pred_cls = torch.sigmoid(prediction[..., 5:])

        if self.training:
            bboxes = torch.cat((xy, wh), 4)
            output = torch.cat((
                bboxes,
                pred_conf,
                pred_cls
            ), -1)
        else:
            xy = xy + self.grid_xy
            wh = torch.exp(wh) * self.anchor_wh
            bboxes = torch.cat((xy, wh), 4)
            output = torch.cat((
                bboxes * self.stride,
                pred_conf,
                pred_cls
            ), -1).view(batch_size, -1, self.n_predict)

        return output


class YoloV3Head(nn.Module):
    def __init__(self, img_size: int, backbone_channels: Tuple[int, int, int, int, int], 
                 grids: List[Tuple[int, int]], anchors: List[Tuple[int, int]], 
                 n_classes: int, backbone_stages: Tuple[int, int, int] = (3, 4, 5)):
        super(YoloV3Head, self).__init__()
        # c1, c2, c3, c4, c5 = backbone_channels
        assert len(backbone_channels) == 5 and len(backbone_stages) == 3
        c3, c4, c5 = (backbone_channels[x-1] for x in backbone_stages)

        scale = (5 - backbone_stages[1]) * 2
        self.u1 = YoloV3UpsampleBlock(in_channels=c5, out_channels=c5, scale=scale)
        self.c1 = YoloV3LayerConv(in_channels=c5//2, n_classes=n_classes)
        self.h1 = YoloV3Layer(
            img_size=img_size,
            grids=grids[2], mask=(6, 7, 8),
            n_classes=n_classes,
            anchors=anchors
        )

        scale = (4 - backbone_stages[0]) * 2
        self.u2 = YoloV3UpsampleBlock(in_channels=(c4 + c5 // 4), out_channels=c4, scale=scale)
        self.c2 = YoloV3LayerConv(in_channels=c4//2, n_classes=n_classes)
        self.h2 = YoloV3Layer(
            img_size=img_size,
            grids=grids[1], mask=(3, 4, 5),
            n_classes=n_classes,
            anchors=anchors
        )

        self.u3 = YoloV3ConvBlock(in_channels=(c3 + c4 // 4), out_channels=c3)
        self.c3 = YoloV3LayerConv(in_channels=c3//2, n_classes=n_classes)
        self.h3 = YoloV3Layer(
            img_size=img_size,
            grids=grids[0], mask=(0, 1, 2),
            n_classes=n_classes,
            anchors=anchors
        )

    def get_anchors(self):
        anchor_vecs = [
            self.h1.anchor_vec,
            self.h2.anchor_vec,
            self.h3.anchor_vec,
        ]
        return anchor_vecs

    def get_ref_anchors(self):
        ref_anchors = [
            self.h1.ref_anchors,
            self.h2.ref_anchors,
            self.h3.ref_anchors,
        ]
        return ref_anchors

    def get_anchor_masks(self):
        anchor_masks = [
            self.h1.anchor_mask,
            self.h2.anchor_mask,
            self.h3.anchor_mask,
        ]
        return anchor_masks

    def forward(self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c1, u1 = self.u1(c5, c4)
        c2, u2 = self.u2(u1, c3)
        c3 = self.u3(u2)
        h1 = self.h1(self.c1(c1))
        h2 = self.h2(self.c2(c2))
        h3 = self.h3(self.c3(c3))
        return h1, h2, h3


def get_head(*args, **kwargs):
    return YoloV3Head(*args, **kwargs)
