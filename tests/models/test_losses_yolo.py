import unittest
import torch

from vortex.development.networks.models.detection.yolov3 import YoloV3
from vortex.development.networks.modules.losses.utils.yolov3 import build_targets, encode_grid_labels, encode_yolo_bbox_labels
from vortex.development.networks.modules.losses.yolov3 import YoloV3Loss as YoloLoss

class TestYoLosses(unittest.TestCase) :
    
    def test_encode_labels(self) :
        n_classes = 1
        img_size = 416
        yolov3 = YoloV3(
            backbone='darknet53',
            img_size=img_size,
            n_classes=n_classes
        )
        grids = yolov3.grids
        ## grids = [(52, 52), (26, 26), (13, 13)]
        anchor_vecs = yolov3.get_anchors()
        ## simulate a gt with xywh (0.5,0.5,0.1,0.1) and class 1
        ## this means the bbox are in the center
        ## and then build with fine-grained grid (52,52)
        target = torch.Tensor([[0, 0, 0.45, 0.45, 0.1, 0.1]])
        encoded_targets =  encode_grid_labels(
            det_shape=torch.Size([1,3,grids[0][0]]),
            targets=target,
            anchors=anchor_vecs[0],
            ignore_thresh=0.5,
            device='cpu'
        )
        b, best_n, gx, gy, gw, gh, gi, gj, is_obj_mask, no_obj_mask, target_labels, target_boxes = encoded_targets
        self.assertEqual(gi,26)
        self.assertEqual(gj,26)
        self.assertTrue(torch.all(torch.eq((is_obj_mask > 0).nonzero(),torch.LongTensor([0,1,26,26]))))

    
    def test_encode_yolo_labels(self) :
        n_classes = 1
        img_size = 416
        yolov3 = YoloV3(
            backbone='darknet53',
            img_size=img_size,
            n_classes=n_classes
        )
        grids = yolov3.grids
        ## grids = [(52, 52), (26, 26), (13, 13)]
        anchor_vecs = yolov3.get_anchors()
        ## simulate a gt with xywh (0.5,0.5,0.1,0.1) and class 1
        ## this means the bbox are in the center
        ## and then build with fine-grained grid (52,52)
        target = torch.Tensor([[0, 0, 0.5, 0.5, 0.1, 0.1]])
        encoded_targets = encode_yolo_bbox_labels(
            det_shape=torch.Size([1,3,grids[0][0]]),
            targets=target,
            n_classes=n_classes,
            anchors=anchor_vecs[0],
            ignore_thresh=0.5,
            device='cpu'
        )
        b, best_n, gj, gi, target_boxes, target_labels, is_obj_mask, no_obj_mask, tx, ty, tw, th, tc, tconf = encoded_targets
        self.assertEqual(tx.size(), torch.Size([1,3,52,52]))
        self.assertEqual(ty.size(), torch.Size([1,3,52,52]))
        self.assertEqual(tw.size(), torch.Size([1,3,52,52]))
        self.assertEqual(th.size(), torch.Size([1,3,52,52]))
        self.assertEqual(tc.size(), torch.Size([1,3,52,52,n_classes]))

    def test_1(self) :
        n_classes = 1
        yolov3 = YoloV3(
            backbone='darknet53',
            img_size=416,
            n_classes=n_classes
        )
        grids = yolov3.grids
        anchor_vecs = yolov3.get_anchors()
        anchors = anchor_vecs[-1]
        target = torch.Tensor([[0, 0, 0.5, 0.5, 0.1, 0.1]])
        pred_boxes = torch.rand(1,3,13,13,4)
        pred_cls = torch.Tensor(torch.rand(1,3,13,13,n_classes))
        targets = build_targets(
            pred_shape=(1,3,13),
            pred_cls=pred_cls,
            targets=target,
            anchors=anchors,
            ignore_thresh=0.5,
            device='cpu'
        )
        class_mask, fg_mask, bg_mask, tx, ty, tw, th, tc, tconf = targets
    
    def test_2(self) :
        target = torch.tensor([[0.0000, 3.0000, 0.1997, 0.6813, 0.4039, 0.9854],
            [0.0000, 3.0000, 0.6982, 0.4123, 0.8949, 0.6287],
            [0.0000, 3.0000, 0.6051, 0.3713, 0.7402, 0.5322],
            [0.0000, 0.0000, 0.6276, 0.5380, 0.8483, 0.9912],
            [0.0000, 0.0000, 0.0030, 0.2135, 0.2688, 0.4854],
            [1.0000, 1.0000, 0.5125, 0.1231, 0.5797, 0.2694],
            [1.0000, 1.0000, 0.8156, 0.2176, 0.8771, 0.3852],
            [2.0000, 1.0000, 0.6714, 0.1898, 0.7177, 0.3111],
            [2.0000, 1.0000, 0.5984, 0.1463, 0.6453, 0.2648],
            [2.0000, 1.0000, 0.5682, 0.1333, 0.5969, 0.2306],
            [2.0000, 1.0000, 0.7599, 0.2509, 0.8385, 0.4287],
            [2.0000, 1.0000, 0.8255, 0.3519, 0.8938, 0.5241],
            [2.0000, 1.0000, 0.9021, 0.4324, 0.9974, 0.6815],
            [3.0000, 3.0000, 0.8531, 0.7694, 0.9677, 0.9833],
            [3.0000, 3.0000, 0.7057, 0.1944, 0.7359, 0.2472],
            [3.0000, 3.0000, 0.5443, 0.1787, 0.6010, 0.2315],
            [3.0000, 3.0000, 0.5193, 0.3111, 0.5708, 0.4287],
            [3.0000, 3.0000, 0.9047, 0.4204, 0.9578, 0.5056],
            [3.0000, 0.0000, 0.1609, 0.6370, 0.2828, 0.9102],
            [3.0000, 1.0000, 0.0000, 0.7954, 0.1141, 0.9963],
            [3.0000, 3.0000, 0.7849, 0.2639, 0.8177, 0.3065],
            [3.0000, 2.0000, 0.0922, 0.7139, 0.1625, 0.8185],
            [3.0000, 2.0000, 0.2948, 0.6889, 0.3531, 0.7759]]
        )
        n_classes = 4
        yolov3 = YoloV3(
            backbone='darknet53',
            img_size=416,
            n_classes=n_classes
        )
        grids = yolov3.grids
        anchor_vecs = yolov3.get_anchors()
        pred_cls = torch.Tensor(torch.rand(1,3,13,13,n_classes))
        dets = yolov3(torch.rand(4,3,416,416))
        criterion = YoloLoss(weight_fg=0.9,weight_bg=0.1,ignore_thresh=0.5)
        criterion.assign_anchors(anchor_vecs)
        loss = criterion(dets,target)
    
    def test_yolo_loss(self) :
        yolov3 = YoloV3(
            backbone='darknet53',
            img_size=416,
            n_classes=1
        )
        anchors = yolov3.get_anchors()
        dets = yolov3(torch.rand(1,3,416,416))
        criterion = YoloLoss(weight_fg=0.9,weight_bg=0.1,ignore_thresh=0.5)
        targets = torch.Tensor([[0, 0, 0.5, 0.5, 0.1, 0.1]])
        criterion.assign_anchors(anchors)
        losses = criterion(dets,targets)
        self.assertTrue(losses.item() > 0)
    
    def gpu_test_yolo_loss(self) :
        device = 'cuda'
        yolov3 = YoloV3(
            backbone='darknet53',
            img_size=416,
            n_classes=1
        ).to(device)
        anchors = yolov3.get_anchors()
        dets = yolov3(torch.rand(1,3,416,416).to(device))
        criterion = YoloLoss(weight_fg=0.9,weight_bg=0.1,ignore_thresh=0.5).to(device)
        criterion.assign_anchors(anchors)
        targets = torch.Tensor([[0, 0, 0.5, 0.5, 0.1, 0.1]]).to(device)
        losses = criterion(dets,targets)
        self.assertTrue(losses.item() > 0)        
