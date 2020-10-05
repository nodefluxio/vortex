import sys
sys.path.insert(0,'src/development')

import torch
import unittest

from vortex.development.networks.models.detection.yolov3 import YoloV3

class YoloV3MobilenetV3Tests(unittest.TestCase) :
    def test_train(self) :
        yolov3 = YoloV3(backbone='mobilenetv3_large_075', img_size=416, n_classes=1)
        out = yolov3(torch.rand(1,3,416,416))
        self.assertEqual(len(out),3)
        self.assertEqual(out[0].size(),torch.Size([1,3,13,13,6]))
        self.assertEqual(out[1].size(),torch.Size([1,3,26,26,6]))
        self.assertEqual(out[2].size(),torch.Size([1,3,52,52,6]))

    def test_train_dynamic(self) :
        yolov3 = YoloV3(backbone='mobilenetv3_large_075', img_size=416, n_classes=1)
        out = yolov3(torch.rand(1,3,608,608))
        self.assertEqual(len(out),3)
        self.assertEqual(out[0].size(),torch.Size([1,3,19,19,6]))
        self.assertEqual(out[1].size(),torch.Size([1,3,38,38,6]))
        self.assertEqual(out[2].size(),torch.Size([1,3,76,76,6]))

    def test_eval(self) :
        yolov3 = YoloV3(backbone='mobilenetv3_large_075', img_size=416, n_classes=1).eval()
        out = yolov3(torch.rand(1,3,416,416))
        self.assertEqual(out.size(),torch.Size([1,3*13*13+3*26*26+3*52*52,6]))

    def gpu_test_train(self) :
        device = torch.device('cuda')
        yolov3 = YoloV3(backbone='mobilenetv3_large_075', img_size=416, n_classes=1).to(device)
        out = yolov3(torch.rand(1,3,416,416).to(device))
        self.assertEqual(len(out),3)
        self.assertEqual(out[0].size(),torch.Size([1,3,13,13,6]))
        self.assertEqual(out[1].size(),torch.Size([1,3,26,26,6]))
        self.assertEqual(out[2].size(),torch.Size([1,3,52,52,6]))

    def gpu_test_train_dynamic(self) :
        device = torch.device('cuda')
        yolov3 = YoloV3(backbone='mobilenetv3_large_075', img_size=416, n_classes=1).to(device)
        out = yolov3(torch.rand(1,3,608,608).to(device))
        self.assertEqual(len(out),3)
        self.assertEqual(out[0].size(),torch.Size([1,3,19,19,6]))
        self.assertEqual(out[1].size(),torch.Size([1,3,38,38,6]))
        self.assertEqual(out[2].size(),torch.Size([1,3,76,76,6]))

    def gpu_test_eval(self) :
        device = torch.device('cuda')
        yolov3 = YoloV3(backbone='mobilenetv3_large_075', img_size=416, n_classes=1).eval().to(device)
        out = yolov3(torch.rand(1,3,416,416).to(device))
        self.assertEqual(out.size(),torch.Size([1,3*13*13+3*26*26+3*52*52,6]))
