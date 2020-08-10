import cv2
import numpy as np
from typing import Tuple, Union, List, Dict, Sequence

colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 127, 127), (127, 255, 127), (127, 127, 255),
    (255, 0, 255), (0, 255, 255), (255, 255, 255),
] * 115 ## >= imagenet

font, font_scale, line_type = cv2.FONT_HERSHEY_SIMPLEX, 1, 2

def draw_bbox(vis: np.ndarray, tl: Tuple[int,int], rb: Tuple[int,int], color: Tuple[int,int,int]=colors[0]) :
    cv2.rectangle(vis, tl, rb, color, 1)
    return vis

def draw_bboxes(vis: np.ndarray, bboxes, classes, confidences, color_map=colors, class_names=None) :
    for bbox, label, confidence in zip(bboxes, classes, confidences) :
        label = int(label)
        color = color_map[label]
        x1, y1, x2, y2 = bbox
        vis = draw_bbox(vis, (x1,y1), (x2,y2), color=color)
        vis = draw_label(vis, label, confidence, (x1,y1), color, class_names=class_names)
    return vis

def draw_landmarks(vis: np.ndarray, landmarks,color: Tuple[int,int,int]=colors[0], radius=2, thickness=-1) :
    for landmark in landmarks :
        assert (len(landmark) % 2) == 0
        xpts, ypts = landmark[0::2], landmark[1::2]
        for x, y in zip(xpts, ypts) :
            pt = int(x), int(y)
            cv2.circle(vis, pt, radius=radius, color=color, thickness=thickness)
    return vis

def draw_label(vis, obj_class, confidence, bl, color, class_names=None) :
    obj_class = obj_class.item() if isinstance(obj_class, np.ndarray) else int(obj_class)
    confidence = confidence.item() if isinstance(confidence, np.ndarray) else float(confidence)
    class_name = class_names[obj_class] if class_names else 'class_{}'.format(obj_class)
    class_name = '{0} : {1:.2f}'.format(class_name, confidence)
    cv2.putText(vis, class_name, bl,
                font, font_scale, color, line_type)
    return vis

def draw_labels(vis, obj_classes, confidences, bls : Sequence[Tuple[int,int]], color_map=colors, class_names=None) :
    for obj_class, confidence, bl in zip(obj_classes, confidences, bls) :
        color = color_map[int(obj_class)]
        vis = draw_label(vis, obj_class, confidence, bl, color, class_names=class_names)
    return vis

def visualize_result(vis : np.ndarray, results : List[Dict[str,np.ndarray]] , class_names=None, color_map=colors) :
    im_h, im_w, im_c = vis.shape
    for result in results :
        assert 'class_label' in result
        class_label = result['class_label']
        class_confidence = result['class_confidence']
        if class_label is None :
            continue
        if 'bounding_box' in result :
            assert 'class_confidence' in result
            bounding_box = result['bounding_box']
            if bounding_box is None :
                continue
            vis = draw_bboxes(vis, bounding_box, class_label, class_confidence, class_names=class_names, color_map=color_map)
        else :
            label_pts = np.asarray([[0, int(im_h*0.95)]]*len(class_label))
            label_pts = [tuple(label_pt) for label_pt in label_pts.tolist()]
            vis = draw_labels(vis, class_label.astype(np.int), class_confidence, label_pts, color_map=color_map, class_names=class_names)
        if 'landmarks' in result :
            landmarks = result['landmarks']
            if landmarks is None :
                continue
            vis = draw_landmarks(vis, landmarks)
    return vis