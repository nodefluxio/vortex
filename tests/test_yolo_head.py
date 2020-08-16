import sys
sys.path.append('src/development')

import unittest
import torch

from vortex.development.networks.modules.utils.darknet import yolo_feature_maps
from vortex.development.networks.modules.heads.detection.yolov3 import YoloV3Layer as YoloLayer
from vortex.development.networks.modules.heads.detection.yolov3 import YoloV3UpsampleBlock as YoloUpsampleBlock
from vortex.development.networks.modules.heads.detection.yolov3 import YoloV3ConvBlock as YoloConvBlock

class YoloHeadTest(unittest.TestCase) :
    def test_yolo_layer_train_1(self):
        mask = [6,7,8]
        anchors = [
            (10,13), (16,30), (33,23), (30,61), (62,45),
            (59,119), (116,90), (156,198), (373,326)
        ]
        n_classes = 1
        img_size = 416
        feature_maps = yolo_feature_maps(img_size)
        grids = feature_maps[-1]
        yolo = YoloLayer(
            img_size=img_size,
            grids=grids,
            mask=mask, 
            anchors=anchors, 
            n_classes=n_classes
        )
        x = yolo(torch.rand([1,18,13,13]))
        self.assertEqual(x.size(),torch.Size([1,3,13,13,6]))
        pred_center = x[:,:,:,:,:2]
        pred_wh = x[:,:,:,2:4]
        pred_obj = x[:,:,:,:,4]
        pred_cls = x[:,:,:,:,5:]
        self.assertTrue(
            torch.all(pred_obj >= 0) and torch.all(pred_obj <= 1)
        )
        self.assertTrue(
            torch.all(pred_cls >= 0) and torch.all(pred_cls <= 1)
        )
        self.assertTrue(
            torch.all(pred_center >= 0) and torch.all(pred_center <= 1)
        )
        self.assertTrue(
            torch.all(torch.exp(pred_wh) < img_size)
        )
    
    def test_yolo_layer_train_2(self) :
        mask = [6,7,8]
        anchors = [
            (10,13), (16,30), (33,23), (30,61), (62,45),
            (59,119), (116,90), (156,198), (373,326)
        ]
        n_classes = 80
        img_size = 416
        feature_maps = yolo_feature_maps(img_size)
        grids = feature_maps[-1]
        head = YoloLayer(
            img_size=img_size,
            grids=grids,
            mask=mask, 
            anchors=anchors, 
            n_classes=n_classes
        )
        x = head(torch.rand([1,255,13,13]))
        self.assertEqual(x.size(),torch.Size([1,3,13,13,85]))
        pred_center = x[:,:,:,:,:2]
        pred_wh = x[:,:,:,2:4]
        pred_obj = x[:,:,:,:,4]
        pred_cls = x[:,:,:,:,5:]
        self.assertTrue(
            torch.all(pred_obj >= 0) and torch.all(pred_obj <= 1)
        )
        self.assertTrue(
            torch.all(pred_cls >= 0) and torch.all(pred_cls <= 1)
        )
        self.assertTrue(
            torch.all(pred_center >= 0) and torch.all(pred_center <= 1)
        )
        self.assertTrue(
            torch.all(torch.exp(pred_wh) < img_size)
        )
    
    def test_yolo_layer_eval_1(self):
        mask = [6,7,8]
        anchors = [
            (10,13), (16,30), (33,23), (30,61), (62,45),
            (59,119), (116,90), (156,198), (373,326)
        ]
        n_classes = 1
        img_size = 416
        feature_maps = yolo_feature_maps(img_size)
        grids = feature_maps[-1]
        yolo = YoloLayer(
            img_size=img_size,
            grids=grids,
            mask=mask, 
            anchors=anchors, 
            n_classes=n_classes
        ).eval()
        x = yolo(torch.rand([1,18,13,13]))
        meshgrid = torch.meshgrid([torch.arange(13), torch.arange(13)]) 
        meshgrid = torch.stack(tuple(reversed(meshgrid)), 2).view(1,13*13,2)
        meshgrid = torch.cat((meshgrid.clone(),meshgrid.clone(),meshgrid.clone()),1)
        self.assertEqual(x.size(),torch.Size([1,3*13*13,6]))
        min_center = (meshgrid + 0) * 32
        max_center = (meshgrid + 1) * 32
        pred_center = x[:,:,:2]
        pred_wh = x[:,:,2:4]
        pred_obj = x[:,:,4]
        pred_cls = x[:,:,5:]
        self.assertTrue(
            torch.all(pred_obj >= 0) and torch.all(pred_obj <= 1)
        )
        self.assertTrue(
            torch.all(pred_cls >= 0) and torch.all(pred_cls <= 1)
        )
        self.assertTrue(
            torch.all(pred_center >= min_center) and torch.all(pred_center <= max_center)
        )
    
    def test_yolo_layer_eval_2(self) :
        mask = [6,7,8]
        anchors = [
            (10,13), (16,30), (33,23), (30,61), (62,45),
            (59,119), (116,90), (156,198), (373,326)
        ]
        n_classes = 80
        img_size = 416
        feature_maps = yolo_feature_maps(img_size)
        grids = feature_maps[-1]
        head = YoloLayer(
            img_size=img_size,
            grids=grids,
            mask=mask, 
            anchors=anchors, 
            n_classes=n_classes
        ).eval()
        x = head(torch.rand([1,255,13,13]))
        meshgrid = torch.meshgrid([torch.arange(13), torch.arange(13)]) 
        meshgrid = torch.stack(tuple(reversed(meshgrid)), 2).view(1,13*13,2)
        meshgrid = torch.cat((meshgrid.clone(),meshgrid.clone(),meshgrid.clone()),1)
        self.assertEqual(x.size(),torch.Size([1,3*13*13,85]))
        min_center = (meshgrid + 0) * 32
        max_center = (meshgrid + 1) * 32
        pred_center = x[:,:,:2]
        pred_wh = x[:,:,2:4]
        pred_obj = x[:,:,4]
        pred_cls = x[:,:,5:]
        self.assertTrue(
            torch.all(pred_obj >= 0) and torch.all(pred_obj <= 1)
        )
        self.assertTrue(
            torch.all(pred_cls >= 0) and torch.all(pred_cls <= 1)
        )
        self.assertTrue(
            torch.all(pred_center >= min_center) and torch.all(pred_center <= max_center)
        )

    def test_yolo_conv_block(self) :
        conv = YoloConvBlock(in_channels=1024)
        x = conv(torch.rand(1,1024,13,13))
        self.assertEqual(x.size(),torch.Size([1,512,13,13]))

    def test_yolo_upsample_block_1(self) :
        upsample = YoloUpsampleBlock(in_channels=1024)
                     # | last stage (#5) |     | route 61 |
        x = upsample(torch.rand(1,1024,13,13),torch.rand(1,512,26,26))
        self.assertEqual(len(x),2)
        self.assertEqual(x[0].size(),torch.Size([1,512,13,13]))
        self.assertEqual(x[1].size(),torch.Size([1,768,26,26]))

    def test_yolo_upsample_block_1(self) :
        upsample = YoloUpsampleBlock(in_channels=768)
                     # | last stage (#5) |     | route 36 |
        x = upsample(torch.rand(1,768,26,26),torch.rand(1,256,52,52))
        self.assertEqual(len(x),2)
        self.assertEqual(x[0].size(),torch.Size([1,384,26,26]))
        self.assertEqual(x[1].size(),torch.Size([1,448,52,52]))
