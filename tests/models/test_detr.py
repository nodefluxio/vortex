import torch
import copy

from vortex.development.networks.models.detection.detr import (
    DETR, HungarianMatcher,
    DETRLoss, DETRPostProcess,
    NestedTensor, cxcywh_to_xyxy
)
from vortex.development.networks.models.detection.detr import DETRCollate


pred_bbox = torch.tensor([
    [[0.2961, 0.5166, 0.2517, 0.6886], [0.0740, 0.8665, 0.1366, 0.1025],
     [0.1841, 0.7264, 0.3153, 0.6871], [0.5756, 0.4966, 0.3164, 0.4017],
     [0.1186, 0.8274, 0.3821, 0.6605], [0.8536, 0.5932, 0.6367, 0.9826],
     [0.2745, 0.6584, 0.2775, 0.8573], [0.8993, 0.0390, 0.9268, 0.7388],
     [0.7179, 0.7058, 0.9156, 0.4340], [0.0772, 0.3565, 0.1479, 0.5331]],

    [[0.4066, 0.2318, 0.4545, 0.9737], [0.4606, 0.5159, 0.4220, 0.5786],
     [0.8455, 0.8057, 0.3775, 0.4087], [0.6179, 0.6932, 0.4354, 0.0353],
     [0.1908, 0.9268, 0.5299, 0.0950], [0.5789, 0.9131, 0.0275, 0.1634],
     [0.3009, 0.2201, 0.2834, 0.3451], [0.0126, 0.7341, 0.9389, 0.8056],
     [0.1459, 0.0969, 0.7076, 0.5112], [0.7050, 0.0114, 0.4702, 0.8526]]
])
pred_logits = torch.tensor([
    [[0.7320, 0.5183, 0.5983, 0.4527, 0.2251], [0.3111, 0.1955, 0.9153, 0.7751, 0.6749],
     [0.1166, 0.8858, 0.6568, 0.8459, 0.3033], [0.6060, 0.9882, 0.8363, 0.9010, 0.3950],
     [0.8809, 0.1084, 0.5432, 0.2185, 0.3834], [0.3720, 0.5374, 0.9551, 0.7475, 0.4979],
     [0.8549, 0.2438, 0.7577, 0.4536, 0.4130], [0.5585, 0.1170, 0.5578, 0.6681, 0.9275],
     [0.3443, 0.6800, 0.9998, 0.2855, 0.9753], [0.2518, 0.7204, 0.6959, 0.6397, 0.8954]],

    [[0.2979, 0.6314, 0.5028, 0.1239, 0.3786], [0.1661, 0.7211, 0.5449, 0.5490, 0.3483],
     [0.5024, 0.3445, 0.6437, 0.9856, 0.5757], [0.2785, 0.1946, 0.5382, 0.1291, 0.1242],
     [0.1746, 0.3302, 0.5370, 0.8443, 0.6937], [0.8831, 0.1861, 0.5422, 0.0556, 0.7868],
     [0.6042, 0.9836, 0.1444, 0.9010, 0.9221], [0.9043, 0.5713, 0.9546, 0.8339, 0.8730],
     [0.4675, 0.1163, 0.4938, 0.5938, 0.1594], [0.2132, 0.0206, 0.3247, 0.9355, 0.5855]]
])

test_outputs = {
    'bbox': pred_bbox, 
    'logits': pred_logits
}

test_targets = [
    {
        'labels': torch.tensor([2, 1]),
        'bbox': torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.2, 0.2, 0.3, 0.3]])
    },
    {
        'labels': torch.tensor([3]),
        'bbox': torch.tensor([[0.8, 0.8, 0.4, 0.4]])
    }
]

num_batch = 2
hidden_dim = 64
num_queries = 10
num_classes = 4


def test_hungarian_matcher():
    expected = [
        torch.tensor([[3, 9], [0, 1]]),
        torch.tensor([[2], [0]])
    ]

    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    indices = matcher(test_outputs, test_targets)
    indices = [torch.stack(x) for x in indices]

    assert all(idx.shape[1] == len(tgt['labels']) for idx, tgt in zip(indices, test_targets))
    assert all(torch.equal(idx, ans) for idx, ans in zip(indices, expected))


def test_detr_loss():
    expected_dict = {
        'loss_ce': torch.tensor(1.5102), 'class_error': torch.tensor(66.6667), 
        'loss_bbox': torch.tensor(0.3037), 'loss_giou': torch.tensor(0.5637), 
        'cardinality_error': torch.tensor(7.5000)
    }
    expected_loss = torch.tensor(4.1563)

    criterion = DETRLoss(num_classes, weight_bbox=5, weight_ce=1, weight_giou=2)
    losses = criterion(test_outputs, test_targets)
    losses_dict = criterion._losses_value

    assert torch.isclose(losses, expected_loss)
    assert all(torch.isclose(expected_dict[n], v, rtol=1e-4, atol=1e-5) for n, v in losses_dict.items())


def test_detr_collater():
    dataformat = {
        'bounding_box': {'indices': [2,3,4,5], 'axis': 1},
        'class_label': {'indices': [1], 'axis': 1}
    }
    collater = DETRCollate(dataformat=dataformat)

    dataset_targets = [
        torch.tensor([[ 0.0000, 14.0000,  0.4604,  0.1122,  0.2292,  0.4439],
                      [ 0.0000, 12.0000,  0.0896,  0.1429,  0.7583,  0.8112],
                      [ 0.0000, 14.0000,  0.1958,  0.3316,  0.0729,  0.1199]]),
        torch.tensor([[ 0.0000, 18.0000,  0.5640,  0.2667,  0.4300,  0.1787]])
    ]
    expected_targets = []
    for target in dataset_targets:
        bbox = target[:, 2:].clone()
        bbox[:, :2] += bbox[:, 2:]/2     # x,y,w,h -> cx,cy,w,h
        labels = target[:, 1].type(torch.int64).clone()
        expected_targets.append({'labels': labels, 'bbox': bbox})

    batch = [
        (torch.randn(3, 800, 900), dataset_targets[0]),
        (torch.randn(3, 800, 1333), dataset_targets[1])
    ]

    images, targets = collater(batch)

    assert isinstance(images, NestedTensor)
    assert all('labels' in t and 'bbox' in t for t in targets)
    assert all(torch.allclose(e['bbox'], t['bbox']) for e,t in zip(expected_targets, targets))
    assert all(torch.equal(e['labels'], t['labels']) for e,t in zip(expected_targets, targets))


def test_detr_postprocess():
    score_threshold = torch.tensor(0.3)
    postprocess = DETRPostProcess()

    outputs_single = torch.cat((pred_bbox[0], pred_logits[0]), -1).unsqueeze(0)
    det = postprocess(outputs_single, score_threshold=score_threshold)
    assert isinstance(det, tuple)
    assert isinstance(det[0], torch.Tensor) and det[0].size(1) == 6

    outputs = torch.cat((pred_bbox, pred_logits), -1)
    det = postprocess(outputs, score_threshold=score_threshold)

    expected = []
    for bbox, logits in zip(copy.deepcopy(pred_bbox), copy.deepcopy(pred_logits)):
        bbox = cxcywh_to_xyxy(bbox)
        conf, labels = logits.softmax(-1)[:, :-1].max(-1)
        res = torch.cat((bbox, conf.unsqueeze(1), labels.float().unsqueeze(1)), -1)[conf > score_threshold]
        expected.append(res)

    assert isinstance(det, tuple)
    assert all((isinstance(d, torch.Tensor) and d.size(1) == 6) for d in det)
    assert all(torch.equal(e, r) for e, r in zip(expected, det))
