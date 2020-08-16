from __future__ import print_function
from __future__ import division

from .tools import is_intersect, intersection, union
from pathlib import Path
from typing import Union, List, Dict

__all__ = [
    'BoundingBox',
]


class BoundingBox:
    def __init__(self, *, x, y, w, h, class_label, img_name, confidence=0, iou=None, tp=None):
        """ the * argument is used to enforce the use of keyword arguments for all arguments
        """
        self.x = float(x)
        self.y = float(y)
        self.w = float(w)
        self.h = float(h)
        self.class_label = class_label
        self.img_name = img_name

        # for detection bbox only
        self.confidence = float(confidence)
        self.iou = iou
        self.tp = tp

    def __str__(self):
        return '[{}] <{}> ({},{},{},{},{})'.format(self.img_name, self.class_label, 
            self.x, self.y, self.w, self.h, self.confidence)

    def __repr__(self):
        return str(self)

    def is_true_positive(self) -> bool:
        return (not self.unmarked()) and self.tp

    def is_false_positive(self) -> bool:
        return (not self.unmarked()) and (not self.tp)

    def unmarked(self) -> bool:
        return self.tp is None

    def mark_tp(self, tp=False, iou=0.):
        self.tp = tp
        self.iou = iou

    def get_xywh(self):
        return (self.x, self.y, self.w, self.h)

    def get_x1y1x2y2(self):
        x1 = float(self.x)
        y1 = float(self.y)
        x2 = x1 + self.w
        y2 = y1 + self.h
        return (x1, y1, x2, y2)

    def get_area(self):
        return self.w * self.h

    def is_intersect(self, other):
        assert isinstance(other, BoundingBox)
        return is_intersect(self.get_x1y1x2y2(), other.get_x1y1x2y2())

    def intersection(self, other) -> float:
        assert isinstance(other, BoundingBox)
        return intersection(self.get_x1y1x2y2(), other.get_x1y1x2y2())

    def union(self, other) -> float:
        assert isinstance(other, BoundingBox)
        return union(self.get_x1y1x2y2(), other.get_x1y1x2y2())
