import sys
sys.path.insert(0,'src/development')

from vortex.development.utils.prediction.tools import *
from vortex.development.utils.prediction import BoundingBox

import pytest


def test_not_intersect():
    box_a = 1., 1., 4., 4.
    box_b = 5., 1., 8., 4.
    x1a, y1a, x2a, y2a = box_a
    x1b, y1b, x2b, y2b = box_b
    assert not is_intersect(box_a, box_b)
    assert not is_intersect(box_b, box_a)
    assert not is_intersect(x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b)
    assert not is_intersect(x1b, y1b, x2b, y2b, x1a, y1a, x2a, y2a)


def test_intersect():
    box_a = 1., 7., 4., 11.
    box_b = 3., 9., 6., 12.
    x1a, y1a, x2a, y2a = box_a
    x1b, y1b, x2b, y2b = box_b
    assert is_intersect(box_a, box_b)
    assert is_intersect(box_b, box_a)
    assert is_intersect(x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b)
    assert is_intersect(x1b, y1b, x2b, y2b, x1a, y1a, x2a, y2a)


def test_intersection():
    box_a = 1., 1., 4., 4.
    box_b = 5., 1., 8., 4.
    box_c = 1., 5., 4., 8.
    box_d = 2., 6., 5., 9.
    box_e = 7., 5., 14., 11.
    box_f = 11., 5., 14., 8.
    assert intersection(box_a, box_a) == 9.
    assert intersection(box_a, box_b) == 0.
    assert intersection(box_c, box_d) == 4.
    assert intersection(box_d, box_c) == 4.
    assert intersection(box_e, box_f) == 9.
    assert intersection(box_f, box_e) == 9.


def test_union():
    box_a = 1., 1., 4., 4.
    box_b = 5., 1., 8., 4.
    box_c = 1., 5., 4., 8.
    box_d = 2., 6., 5., 9.
    box_e = 7., 5., 14., 12.
    box_f = 11., 5., 14., 8.
    assert union(box_a, box_a) == 9.
    assert union(box_a, box_b) == 18.
    assert union(box_c, box_d) == 14.
    assert union(box_d, box_c) == 14.
    assert union(box_e, box_f) == 49.
    assert union(box_f, box_e) == 49.


def test_exception() :
    box_1 = 1., 2.,
    box_2 = 1., 2.,
    box_3 = 1., 2., 3.
    box_4 = 1., 2., 3.
    with pytest.raises(RuntimeError):
        is_intersect(box_1, box_2)
        is_intersect(box_3, box_4)
        intersection(box_1, box_2)
        intersection(box_3, box_4)
        union(box_1, box_2)
        union(box_3, box_4)
        area(box_1, box_2)
        area(box_3, box_4)


def test_bbox():
    box_a = {'x': 1., 'y': 1., 'w': 4., 'h': 4.}
    box_a_xyxy = {'x': 1., 'y': 1., 'w': 5., 'h': 5.}
    bbox_a = BoundingBox(**box_a, class_label=0, img_name=1)

    assert tuple(box_a.values()) == bbox_a.get_xywh()
    assert tuple(box_a_xyxy.values()) == bbox_a.get_x1y1x2y2()
    assert bbox_a.get_area() == box_a['w'] * box_a['h']

def test_bbox_detection():
    box_a = {'x': 1, 'y': 5, 'w': 3, 'h': 3}
    box_b = {'x': 2, 'y': 6, 'w': 3, 'h': 3}
    bbox_a = BoundingBox(**box_a, class_label=0, img_name=1)
    bbox_b = BoundingBox(**box_b, class_label=1, img_name=2)
    assert bbox_a.unmarked()
    assert bbox_b.unmarked()

    bbox_a.mark_tp(tp=True, iou=0.2)
    assert bbox_a.is_true_positive()
    assert not bbox_a.is_false_positive()
    
    assert bbox_a.is_intersect(bbox_b)
    assert bbox_b.is_intersect(bbox_a)
    assert bbox_a.intersection(bbox_b) == 4.
    assert bbox_b.intersection(bbox_a) == 4.
    assert bbox_a.union(bbox_b) == 14.
    assert bbox_b.union(bbox_a) == 14.
