import sys
sys.path.append('vortex/development_package')

from vortex.development.utils.metrics.evaluator import *
from vortex.development.utils.prediction.bboxes import BoundingBox


bbox_a = BoundingBox(x=1., y=1., w=4., h=4., class_label=0, img_name='test.jpg')
bbox_b = BoundingBox(x=5., y=1., w=8., h=4., class_label=0, img_name='test.jpg')
bbox_c = BoundingBox(x=1., y=5., w=4., h=8., class_label=0, img_name='test.jpg')
bbox_d = BoundingBox(x=2., y=6., w=5., h=9., class_label=0, img_name='test.jpg')
bbox_e = BoundingBox(x=7., y=5., w=14., h=12., class_label=0, img_name='test.jpg')
bbox_f = BoundingBox(x=11., y=5., w=14., h=8., class_label=0, img_name='test.jpg')

def test_best_match():
    labels = [bbox_a, bbox_b, bbox_c, bbox_d, bbox_e, bbox_f]
    detections = [bbox_a, bbox_b, bbox_c, bbox_d, bbox_e, bbox_f]
    
    for i, label in enumerate(labels) :
        match_idx, match_iou = best_match(label,detections)
        assert match_idx == i
        assert match_iou == 1
    matches = best_match(labels[0],detections[1:])
    match_idx, match_iou = matches
    assert match_idx == None
    assert match_iou == None

def test_mark_bbox():
    labels = [bbox_a, bbox_b, bbox_c, bbox_d, bbox_e, bbox_f]
    detections = [bbox_a, bbox_b, bbox_c, bbox_d, bbox_e, bbox_f]

    marked = mark_bbox(labels, detections)
    assert len(marked) == 6
    assert all([bbox.is_true_positive() for bbox in marked])

    marked = mark_bbox(labels[1:], detections)
    assert len(marked) == 6
    assert [bbox.is_true_positive() for bbox in marked].count(True) == 5
    assert [bbox.is_false_positive() for bbox in marked].count(True) == 1

    marked = mark_bbox(labels,detections[1:])
    assert len(marked) == 5
    assert all([bbox.is_true_positive() for bbox in marked])


def test_mark_bboxes():
    labels = [bbox_a, bbox_b, bbox_c, bbox_d, bbox_e, bbox_f]
    detections = [bbox_a, bbox_b, bbox_c, bbox_d, bbox_e, bbox_f]
    gt_bboxes = BoundingBoxes() + labels
    det_bboxes = BoundingBoxes() + detections

    marked_bboxes = mark_bboxes(det_bboxes, gt_bboxes)
    assert len(marked_bboxes) == 6
    assert all([bbox.is_true_positive() for bbox in marked_bboxes])


def test_tp_fp():
    detections = [bbox_a, bbox_b, bbox_c, bbox_d, bbox_e, bbox_f]
    assert all([det.unmarked() for det in detections])

    detections[0].mark_tp(tp=True)
    detections[3].mark_tp(tp=True)
    assert len([box for box in detections if box.is_true_positive()]) == 2
    assert len([box for box in detections if box.unmarked()]) == 4

    detections[4].mark_tp(tp=False)
    assert len([box for box in detections if box.is_false_positive()]) == 1


def test_eleven_point_interpolation() :
    pr_list = [(1.0,1.0) for i in range(11)]
    ap = eleven_point_interpolation(pr_list)
    assert ap == 1.0

    pr_list = [(0.5,1.0) for i in range(100)]
    ap = eleven_point_interpolation(pr_list)
    assert ap == 0.5

    pr_list = [(0.5,1.0) for i in range(5)]
    pr_list = pr_list + [(0.7,0.5) for i in range(5)]
    ap = eleven_point_interpolation(pr_list)
    """
    inter_pt precision ap
    0.0 0.7 0.7
    0.1 0.7 1.4
    0.2 0.7 2.1
    0.3 0.7 2.8
    0.4 0.7 3.5
    0.5 0.7 4.2
    0.6 0.5 4.7
    0.7 0.5 5.2
    0.8 0.5 5.7
    0.9 0.5 6.2
    1.0 0.5 6.7
    in short : 0.7*5 + 0.5*5 = 6.7
    """
    assert (6.7/11 - 1e-6) < ap < (6.7/11 + 1e-6)

    pr_list = [(0.5,1.0) for i in range(5)]
    pr_list = pr_list + [(0.7,0.5) for i in range(5)]
    pr_list = pr_list + [(0.9,0.25) for i in range(3)]
    ap = eleven_point_interpolation(pr_list)
    expected_ap = (0.9*3 + 0.7*3 + 0.5*5) / 11.
    assert (expected_ap - 1e-6) < ap < (expected_ap + 1e-6)

def test_class_ap():
    labels = [bbox_a, bbox_b, bbox_c, bbox_d, bbox_e, bbox_f]
    detections = [bbox_a, bbox_b, bbox_c, bbox_d, bbox_e, bbox_f]
    gt_bboxes = BoundingBoxes() + labels
    det_bboxes = BoundingBoxes() + detections
    detections = mark_bboxes(det_bboxes, gt_bboxes).filter(class_label=0)
    ap, precision, recall = class_average_precision(detections, labels)

    assert ap == 1.0
    assert precision[-1] == 1.0
    assert recall[-1] == 1.0


def test_ap():
    labels = [bbox_a, bbox_b, bbox_c, bbox_d, bbox_e, bbox_f]
    detections = [bbox_a, bbox_b, bbox_c, bbox_d, bbox_e, bbox_f]
    gt_bboxes = BoundingBoxes() + labels
    det_bboxes = BoundingBoxes() + detections
    marked_bboxes = mark_bboxes(det_bboxes, gt_bboxes)
    result = average_precision(marked_bboxes,gt_bboxes)
    assert len(result) == 2
    assert result[1] == 1.0

    result = result[0][0]
    ap = result['ap']
    precision = result['precision']
    recall = result['recall']
    assert ap == 1.0
    assert precision[-1] == 1.0
    assert recall[-1] == 1.0

def test_evaluator():
    labels = [bbox_a, bbox_b, bbox_c, bbox_d, bbox_e, bbox_f]
    detections = [bbox_a, bbox_b, bbox_c, bbox_d, bbox_e, bbox_f]

    evaluator = DetectionEvaluator()
    for label, detection in zip(labels[:3],detections[:3]) :
        evaluator.update(detection,label)
    evaluator.update(detections[3:],labels[3:])
    assert evaluator.n_items() == (6,6)
    assert evaluator.n_images() == (1,1)

    eval_result = evaluator.evaluate()
    assert len(eval_result) == 2
    assert eval_result[1] == 1.0

    eval_result = eval_result[0][0]
    ap = eval_result['ap']
    precision = eval_result['precision']
    recall = eval_result['recall']
    assert ap == 1.0
    assert precision[-1] == 1.0
    assert recall[-1] == 1.0
