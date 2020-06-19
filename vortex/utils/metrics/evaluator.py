from vortex.utils.prediction.bboxes import BoundingBox
from typing import Union, List, Dict, Tuple
from copy import copy

import numpy as np
import matplotlib.pyplot as plt
import multipledispatch
import warnings
import enforce

__all__ = [
    'BoundingBoxes',
    'DetectionEvaluator',
    'best_match',
    'mark_bbox',
    'mark_bboxes',
    'unmarked',
    'eleven_point_interpolation',
    'class_average_precision',
    'average_precision',
]


@enforce.runtime_validation
def unmarked(bboxes: List[BoundingBox]) -> List[BoundingBox]:
    return [box for box in bboxes if box.unmarked()]


class BoundingBoxes(object):
    def __init__(self):
        self.bbox_list = []
        self.class_map = {}
        self.n_classes = 0
        self.img_names = []

    @property
    def classes(self):
        return self.n_classes

    @property
    def filenames(self):
        return self.img_names

    def distribution(self, class_label: int, key='area', img_root='.'):
        dist = []
        for filename in self.filenames:
            bboxes = self.filter_by_class_and_img_name(class_label, filename)
            filename = filename.rstrip().replace('# ', '')
            img = cv2.imread(str(Path(img_root)/Path(filename)))
            shape = img.shape
            total_area = (shape[0]*shape[1])
            for box in bboxes:
                for bbox in bboxes:
                    if key == 'area':
                        dist.append(box.get_area()/total_area)
        return dist

    def __len__(self):
        return len(self.bbox_list)

    def __iter__(self):
        self.counter = -1
        return self

    def __str__(self):
        return str(self.bbox_list)

    def __repr__(self):
        return str(self)

    def __getitem__(self, i):
        return self.bbox_list[i]

    def __next__(self):
        self.counter = self.counter + 1
        if self.counter < len(self):
            return self.bbox_list[self.counter]
        else:
            raise StopIteration

    def add_box(self, box: BoundingBox):
        assert(isinstance(box, BoundingBox))
        if box.img_name not in self.img_names:
            self.img_names.append(box.img_name)
        self.bbox_list.append(box)
        if not (box.class_label in self.class_map.values()):
            self.class_map[self.n_classes] = box.class_label
            self.n_classes += 1

    def __add__(self, bboxes):
        self.add_boxes(bboxes)
        return self

    def add_boxes(self, boxes):
        for box in boxes:
            self.add_box(box)

    def get_classes(self):
        return list(self.class_map.values())

    def get_img_names_from_class(self, class_label):
        img_names = set()
        for bbox in self.bbox_list:
            if bbox.class_label == class_label:
                img_names.add(bbox.img_name)
        return list(img_names)

    def filter(self, class_label=None, img_name=None):
        filtered = []
        for bbox in self.bbox_list:
            class_check = (class_label is None) or (
                bbox.class_label == class_label)
            img_check = (img_name is None) or (bbox.img_name == img_name)
            if class_check and img_check:
                filtered.append(bbox)
        return filtered


# TODO : consider returning none instead of tuple(None,None) if no match
def best_match(label: BoundingBox, detections: List[BoundingBox]) -> Tuple[Union[int, None], Union[float, None]]:
    max_iou, detect_idx = None, None
    for i, detection in enumerate(detections):
        intersection = label.intersection(detection)
        union = label.union(detection)
        iou = intersection / union
        if not iou:
            continue
        if (max_iou is None) or (iou > max_iou):
            max_iou = iou
            detect_idx = i
    return detect_idx, max_iou


def mark_bbox(labels: List[BoundingBox], detections: List[BoundingBox], iou_threshold: float = 0.5) -> List[BoundingBox]:
    marked_bboxes = []
    marked_indexes = []
    for i, label in enumerate(labels):
        best_matched = best_match(label, detections)
        best_idx, best_iou = best_matched
        if best_idx is None:
            continue
        elif best_iou > iou_threshold:
            marked_indexes.append(best_idx)
            marked_bbox = copy(detections[best_idx])
            marked_bbox.mark_tp(tp=True, iou=best_iou)
            marked_bboxes.append(marked_bbox)
    unmarked_index = [i for i in range(
        len(detections)) if i not in marked_indexes]
    for i in unmarked_index:
        fp_bbox = copy(detections[i])
        fp_bbox.mark_tp(tp=False)
        marked_bboxes.append(fp_bbox)
    return marked_bboxes


def mark_bboxes(detection_bboxes: BoundingBoxes, gt: BoundingBoxes, iou_threshold: float = 0.5):
    """given detection bboxes, compare it with gt 
    and retrun bounding boxes with marked tp/fp 
    """
    class_map = detection_bboxes.class_map
    marked_bboxes = BoundingBoxes()
    # for each class we find best iou for each image
    for class_ in gt.get_classes():
        # n = len(gt.get_img_names_from_class(class_))
        # for each image, we find detections with bes iou
        for i, img_name in enumerate(gt.get_img_names_from_class(class_)):
            labels = gt.filter(class_label=class_, img_name=img_name)
            detections = detection_bboxes.filter(class_label=class_, img_name=img_name)
            marked_bboxes_class = mark_bbox(labels, detections, iou_threshold)
            marked_bboxes.add_boxes(marked_bboxes_class)
    return marked_bboxes


def eleven_point_interpolation(pr_list: List[List[Tuple[float, float]]]):
    ap = 0
    for pt in [0.1*i for i in range(0, 11)]:
        prec_list = [
            prec for prec, recall in pr_list if recall >= pt
        ]
        precision = max(prec_list) if len(prec_list) else 0.
        ap += precision
        # print(pt,precision,ap)
    ap /= 11
    return ap


def class_average_precision(detections: List[BoundingBox], labels: List[BoundingBox]):
    if any(unmarked(detections)):
        warnings.warn("it seems your detection bboxes has unmarked tp/fp, please check")
    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))
    n_labels = len(labels)
    for i, det in enumerate(detections):
        tp[i] = int(det.is_true_positive())
        fp[i] = int(det.is_false_positive())
        acc_tp_tmp = np.cumsum(tp)
        acc_fp_tmp = np.cumsum(fp)
        recall_tmp = acc_tp_tmp / n_labels
        precision_tmp = np.divide(acc_tp_tmp, (acc_fp_tmp + acc_tp_tmp))
    acc_tp = np.cumsum(tp)
    acc_fp = np.cumsum(fp)
    # print(acc_fp, acc_tp)
    recall = acc_tp / n_labels
    precision = acc_tp / (acc_fp + acc_tp)
    pr_list = [(prec, rec) for prec, rec in zip(precision, recall)]
    ap = eleven_point_interpolation(pr_list)
    return ap, precision, recall


def average_precision(detection_bboxes: BoundingBoxes, gt: BoundingBoxes, 
        iou_threshold: float = 0.5) -> List[Dict[str, Union[float, int]]]:
    classes = detection_bboxes.get_classes()
    results = []
    for class_ in classes:
        detections = detection_bboxes.filter(class_label=class_)
        detections = sorted(detections, key=lambda bbox: (
            bbox.confidence, bbox.is_true_positive()
        ), reverse=True)
        labels = gt.filter(class_label=class_)
        if any(unmarked(detections)):
            raise RuntimeError(
                "it seems your detection bboxes has unmarked tp/fp, please check")
        class_ap, class_prec, class_recall = class_average_precision(
            detections, labels)
        results.append({
            "class": class_,
            "precision": class_prec,
            "recall": class_recall,
            "ap": class_ap,
        })
    mean_ap = 0.
    for result in results:
        mean_ap += result['ap']
    return results, mean_ap


class DetectionEvaluator:
    """
    The Evaluator class, supports for incrementally update bbox & labels
    """

    def __init__(self, iou_threshold: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bboxes = BoundingBoxes()
        self.gt = BoundingBoxes()
        self.iou_threshold = iou_threshold

    def n_items(self):
        return len(self.bboxes), len(self.gt)

    def n_images(self):
        return len(self.bboxes.img_names), len(self.gt.img_names)

    def add_detection(self, bbox: BoundingBox):
        self.bboxes.add_box(bbox)

    def add_label(self, gt: BoundingBox):
        self.gt.add_box(gt)

    @multipledispatch.dispatch(BoundingBoxes, BoundingBoxes)
    def update(self, bboxes: BoundingBoxes, labels: BoundingBoxes):
        self.bboxes.add_boxes(bboxes)
        self.gt.add_boxes(labels)

    @multipledispatch.dispatch(BoundingBox, BoundingBox)
    def update(self, bbox: BoundingBox, label: BoundingBox):
        self.bboxes.add_box(bbox)
        self.gt.add_box(label)

    @multipledispatch.dispatch(list, list)
    def update(self, bboxes: List[BoundingBoxes], labels: List[BoundingBoxes]):
        for bbox in bboxes:
            self.bboxes.add_box(bbox)
        for label in labels:
            self.gt.add_box(label)

    def evaluate(self, iou_threshold: Union[float, None] = None) -> List[Dict[str, Union[float, int]]]:
        if iou_threshold is None:
            iou_threshold = self.iou_threshold
        marked_bbox = mark_bboxes(self.bboxes, self.gt, iou_threshold)
        results, mean_ap = average_precision(marked_bbox, self.gt, iou_threshold)
        # class_map = self.bboxes.class_map
        mean_ap /= self.gt.n_classes
        return results, mean_ap
