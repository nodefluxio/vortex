import cv2
import torch
import numpy as np
from easydict import EasyDict
from typing import List, Dict, Tuple, Union, Any

from albumentations.pytorch import ToTensor
import albumentations.core.composition as albumentations_compose

import albumentations.augmentations.transforms as albumentations_tf

supported_transforms = [
    'albumentations'
]

ACCEPTED_BBOX_PARAMS = ['min_visibility', 'min_area']


class AlbumentationsWrapper:

    def __init__(self, transforms: List[EasyDict], data_format: EasyDict, bbox_params: EasyDict = None, visual_debug: bool = False):
        # Parse augmentations albumentation config
        self.data_format = data_format
        self.visual_debug = visual_debug
        if 'bounding_box' in self.data_format:
            # Prepare labels assocication between bounding box and other labels
            label_fields = ['bbox_local_info']

            # Parse BboxParams
            if bbox_params is not None:
                if any([key not in ACCEPTED_BBOX_PARAMS for key in bbox_params.keys()]):
                    raise RuntimeError("In this albumentations implementation, 'bbox_params' allowed to be modified are %s !! Found %s" % (
                        ACCEPTED_BBOX_PARAMS, bbox_params.keys()))

                if 'min_area' in bbox_params:
                    min_area = bbox_params.min_area
                else:
                    min_area = 0.0
                if 'min_visibility' in bbox_params:
                    min_visibility = bbox_params.min_visibility
                else:
                    min_visibility = 0.0

                bbox_params = albumentations_compose.BboxParams(
                    format='coco',
                    label_fields=label_fields,
                    min_area=min_area,
                    min_visibility=min_visibility
                )
            else:
                bbox_params = albumentations_compose.BboxParams(
                    format='coco',
                    label_fields=label_fields,
                )

        # Landmarks Params, remove invisible not allowed to be True because can cause change of shape and make system broken
        landmarks_params = None
        if 'landmarks' in self.data_format:
            # If bounding_box exist, automatically assume landmarks is associated with bounding box
            landmarks_params = albumentations_compose.KeypointParams(
                format='xy',
                remove_invisible=False
            )
        # Parse compose from config
        transforms_compose = _parse_compose(
            transforms)
        self.compose = albumentations_compose.ReplayCompose(
            transforms_compose, bbox_params=bbox_params, keypoint_params=landmarks_params)

    def __call__(self, image: np.ndarray, targets: np.ndarray):
        df = self.data_format
        # Prepare returned template
        ret_targets = targets.copy()
        albu_input = tensor_to_albu_input(image, targets, df)
        if self.visual_debug:
            # VIZ DEBUG
            vis_image = image.copy()
            try:
                vis_landmarks = albu_input['keypoints'].reshape((len(albu_input['bboxes']), int(
                    albu_input['keypoints'].shape[0]/len(albu_input['bboxes'])), 2))
            except:
                pass
            try:
                for i, bbox in enumerate(albu_input['bboxes']):
                    x = int(bbox[0])
                    y = int(bbox[1])
                    w = int(bbox[2])
                    h = int(bbox[3])

                    cv2.rectangle(vis_image, (x, y),
                                  (x+w, y+h), (0, 0, 255), 2)

                    try:
                        labels = albu_input['bbox_local_info'][i][1:]
                        if len(labels) == 0:
                            labels = [0]
                        cv2.putText(
                            vis_image, str(labels), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    except:
                        pass
                    try:
                        object_landmarks = vis_landmarks[i]
                        color = (0, 0, 255)
                        for i, landmark in enumerate(object_landmarks):
                            if 'asymm_pairs' in df.landmarks:
                                asymm_pairs = np.array(
                                    df.landmarks.asymm_pairs)
                                left_group = asymm_pairs[:, 0]
                                if i in left_group:
                                    color = (0, 255, 0)
                                else:
                                    color = (0, 0, 255)
                            cv2.circle(vis_image, (int(landmark[0]), int(
                                landmark[1])), 2, color, -1)
                    except:
                        pass
            except:
                pass

            cv2.imshow('ori', vis_image)
            # VIZ DEBUG

        # Augment data
        annotations = self.compose(**albu_input)

        # Parse result back to tensor
        image = annotations['image']
        ret_targets = albu_result_to_tensor(ret_targets,annotations, albu_input, df)

        if self.visual_debug:
            h, w, c = image.shape
            # VIZ DEBUG
            if 'bounding_box' in df:
                box_indices, box_axis = df.bounding_box.indices, df.bounding_box.axis
                allbboxes = np.take(ret_targets, axis=box_axis,
                                    indices=box_indices)
                allbboxes[:, [0, 2]] *= w
                allbboxes[:, [1, 3]] *= h

            if 'class_label' in df:
                cls_indices, cls_axis = df.class_label.indices, df.class_label.axis
                class_labels = np.take(
                    ret_targets, axis=cls_axis, indices=cls_indices)

            if 'landmarks' in df:
                lmk_indices, lmk_axis = df.landmarks.indices, df.landmarks.axis
                landmarks = np.take(
                    ret_targets, axis=lmk_axis,
                    indices=lmk_indices
                )
            vis_image = image.copy()

            try:
                for i, bbox in enumerate(allbboxes):
                    x = int(bbox[0])
                    y = int(bbox[1])
                    box_w = int(bbox[2])
                    box_h = int(bbox[3])

                    try:
                        cv2.putText(vis_image, str(class_labels[i]), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 0, 0), 2, cv2.LINE_AA)
                    except:
                        cv2.putText(vis_image, str([0]), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 0, 0), 2, cv2.LINE_AA)
                    try:
                        vis_landmarks = landmarks[i].reshape(
                            (int(landmarks[i].size/2), 2))
                        color = (0, 0, 255)
                        for i, landmark in enumerate(vis_landmarks):
                            if 'asymm_pairs' in df.landmarks:
                                asymm_pairs = np.array(
                                    df.landmarks.asymm_pairs)
                                left_group = asymm_pairs[:, 0]
                                if i in left_group:
                                    color = (0, 255, 0)
                                else:
                                    color = (0, 0, 255)
                            cv2.circle(vis_image, (int(landmark[0]*w), int(
                                landmark[1]*h)), 2, color, -1)
                    except:
                        pass

                    cv2.rectangle(vis_image, (x, y),
                                  (x+box_w, y+box_h), (0, 0, 255), 2)

            except:
                pass
            cv2.imshow('aug', vis_image)
            cv2.waitKey(0)
            # VIZ DEBUG

        return image, ret_targets


def _parse_sequential_replay(transforms_replay: List[Dict]):
    """Parse augmentations sequence which is random applied each step to track difference
    """
    sequential_applied = []
    for act in transforms_replay:
        if act['applied']:
            if act['__class_fullname__'].split('.')[2] == 'composition':
                transforms_list = _parse_sequential_replay(
                    act['transforms'])
                for act in transforms_list:
                    sequential_applied.append(act)
            else:
                sequential_applied.append(act)
    return sequential_applied


def _parse_compose(compose: List[EasyDict]):
    '''
    Parse config augmentation composition recursively if found any 'compose' component like 'Compose' and 'OneOf' in the config. 'compose' component must have 'transforms' param in their 'args'. However, 'transform' component didn't have to. If 'args' is not provided in the 'transform' component, then it will use default albumentations value
    '''
    transforms_compose = []
    for component in compose:
        if 'compose' in component.keys():
            if 'args' not in component.keys():
                raise RuntimeError(
                    "compose '%s' albumentations module component must have 'args' described, please check" % component['compose'])
            compose, args = component['compose'], component['args']
            if not compose in albumentations_compose.__all__:
                raise RuntimeError(
                    "unsupported composition '%s' in albumentation please check" % compose)
                if compose not in ['OneOf']:
                    raise RuntimeError(
                        "unsupported composition '%s' in this implementation, only support 'OneOf' composition please check" % compose)
            if 'transforms' not in args.keys():
                raise RuntimeError(
                    "expect 'transforms' param in composition '%s' 'args'" % compose)
            if not isinstance(args['transforms'], list):
                raise RuntimeError(
                    "expect 'transforms' param in composition '%s' 'args' to have type of list got %s" % type(args['transforms']))
            if len(args['transforms']) == 0:
                raise RuntimeError(
                    "'transforms' param in composition '%s' 'args' cannot contain empty list, please check")
            if not all([type(transform) == EasyDict for transform in args['transforms']]):
                raise RuntimeError(
                    "expects type of value of `transforms` to have type of dict got %s" % [type(transform) for transform in args['transforms']])
            args['transforms'] = _parse_compose(
                args['transforms'])
            try:
                transforms_compose.append(
                    albumentations_compose.__getattribute__(compose)(**args))
            except Exception as e:
                raise RuntimeError(
                    'when trying to create instance of %s with args : %s; got the following error : %s; please check albumentations documentations' % (compose, args, str(e)))
        elif 'transform' in component.keys():
            tf = component['transform']
            args = {}
            if 'args' in component.keys():
                args = component['args']
            if not tf in albumentations_tf.__all__:
                raise RuntimeError(
                    'unsupported transform "%s" in albumentation please check' % tf)
            try:
                transforms_compose.append(
                    albumentations_tf.__getattribute__(tf)(**args))
            except Exception as e:
                raise RuntimeError(
                    'when trying to create instance of %s with args : %s; got the following error : %s; please check albumentations documentations' % (tf, args, str(e)))

    return transforms_compose

# Workaround to check coordinates which more than image shape
def fix_bbox_coords(bboxes : np.ndarray,image_shape : tuple):
    bboxes_xymax =bboxes[:,0:2] + bboxes[:,2:4]
    bboxes_xmax = bboxes_xymax[:,0]
    bboxes_ymax = bboxes_xymax[:,1]
    bboxes_xmax[np.where(bboxes_xmax >= image_shape[1] - 3)] = image_shape[1] - 3
    bboxes_ymax[np.where(bboxes_ymax >= image_shape[0] - 3)] = image_shape[0] - 3
    bboxes_xymax[:,0] = bboxes_xmax
    bboxes_xymax[:,1] = bboxes_ymax
    bboxes[:,2:4] = bboxes_xymax - bboxes[:,0:2]
    return bboxes

def tensor_to_albu_input(image: np.ndarray, targets: np.ndarray, data_format: EasyDict):
    h, w, c = image.shape
    # Insert image into input augmentation data
    data = EasyDict({'image': image})
    # Prepare bounding box annotations augmentation
    if 'bounding_box' in data_format:
        box_indices, box_axis = data_format.bounding_box.indices, data_format.bounding_box.axis
        allbboxes = np.take(targets, axis=box_axis, indices=box_indices)
        # Convert to absolute coords related to albumentations 'coco' format 'xywh'
        allbboxes[:, [0, 2]] *= w
        allbboxes[:, [1, 3]] *= h
        allbboxes = fix_bbox_coords(allbboxes,image.shape)
        # Insert bboxes into input augmentation data
        data.bboxes = allbboxes
        # Build local index of objects in an image
        bbox_local_info = np.arange(len(allbboxes)).reshape(-1, 1)

        # Prepare class_label if set
        if 'class_label' in data_format and data_format.class_label:
            cls_indices, cls_axis = data_format.class_label.indices, data_format.class_label.axis
            class_labels = np.take(targets, axis=cls_axis, indices=cls_indices)
            # Due to albumentations multi label fields bug, currently class_labels is concatenated with
            # bbox_local_info in the indice 1, axis 1
            bbox_local_info = np.concatenate((bbox_local_info, class_labels), axis=1)
        # Insert bbox_local_info (bbox local index in an image and/or class_labels) into input augmentation data
        data.bbox_local_info = bbox_local_info
    if 'landmarks' in data_format:
        # Slice landmarks coordinates
        lmk_indices, lmk_axis = data_format.landmarks.indices, data_format.landmarks.axis
        landmarks = np.take(
            targets, axis=lmk_axis,
            indices=lmk_indices
        )
        # Maintain original number of objects, reshape landmarks to albumentation format
        reshaped_landmarks = landmarks.reshape((int(landmarks.size/2), 2))

        # Convert to absolute coordinates
        reshaped_landmarks[:, 0] *= w
        reshaped_landmarks[:, 1] *= h

        data.keypoints = reshaped_landmarks
    return data


def albu_result_to_tensor(ret_targets: np.ndarray, annotations: Dict, albu_input: Dict, data_format: EasyDict):
    image = annotations['image'].copy()
    h, w, c = image.shape
    # Parse augmentation result
    if 'bounding_box' in data_format:
        bboxes = np.array(annotations['bboxes'])
        box_indices, box_axis = data_format.bounding_box.indices, data_format.bounding_box.axis

        # Return 0 result if augment return 0 bboxes annotation
        if len(bboxes) == 0:
            ret_targets = np.empty(
                [0, ret_targets.shape[1]], dtype='float32')
            return ret_targets

        # Convert back from absolute coords to relative coords
        bboxes[:, [0, 2]] /= w
        bboxes[:, [1, 3]] /= h
        mod_len = len(bboxes)
        # Inspect if there are any missing bounding box, reconstruct ret_targets shape alongside local object sequence
        ori_len = len(albu_input['bboxes'])
        bbox_local_info = np.array(annotations['bbox_local_info'])
        if ori_len != mod_len:
            bbox_local_idx = np.take(
                bbox_local_info, axis=1, indices=0).astype('int')
            ret_targets = ret_targets[bbox_local_idx]
        # Update returned target tensor with modified bboxes
        np.put_along_axis(ret_targets, values=bboxes, axis=box_axis,
                          indices=np.array(box_indices)[np.newaxis, :])
        if data_format.class_label:
            cls_indices, cls_axis = data_format.class_label.indices, data_format.class_label.axis
            # Due to albumentations multi label fields bug, currently class_labels is concatenated with
            # bbox_local_info in the indice 1, axis 1
            class_labels = np.take(bbox_local_info, axis=1, indices=[1])

            # Update returned target tensor with modified class_labels related to bboxes
            np.put_along_axis(ret_targets, values=class_labels, axis=cls_axis,
                              indices=np.array(cls_indices)[np.newaxis, :])
    if 'landmarks' in data_format:
        # Obtain modified landmarks
        landmarks = np.array(annotations['keypoints'])
        lmk_indices, lmk_axis = data_format.landmarks.indices, data_format.landmarks.axis

        # Convert to relative coordinates
        landmarks[:, 0] /= w
        landmarks[:, 1] /= h

        # Reshape landmarks from albumentations format into original format
        landmarks = landmarks.reshape(
            (ori_len, int(landmarks.size/ori_len)))

        # if bounding_box exist, automatically assume each set of landmarks is associate with one bounding box
        if 'bounding_box' in data_format:
            if ori_len != mod_len:
                # Remove landmarks labels in which the box is deleted in previous step
                landmarks = landmarks[bbox_local_idx]

        # If any declared asymmetric keypoints pair
        if 'asymm_pairs' in data_format.landmarks:
            # Obtain sequentially applied augmentation
            augments_replay_info = annotations['replay']['transforms']
            augments_seq = _parse_sequential_replay(
                augments_replay_info)
            # Count the appearance of horizontal and vertical flip augmentation which can broke asymmetric keypoints sequential indexing
            hflip_count = len(
                [augment for augment in augments_seq if augment['__class_fullname__'].split('.')[-1] == 'HorizontalFlip'])
            vflip_count = len(
                [augment for augment in augments_seq if augment['__class_fullname__'].split('.')[-1] == 'VerticalFlip'])
            # Apply modification if count is an odd number
            if (hflip_count+vflip_count) % 2 == 1:
                # Shape formatting for better coordinates indexing
                n_keypoints = int(len(landmarks[0])/2)
                landmarks = landmarks.reshape((-1, n_keypoints, 2))

                # For each index keypoints pair, swap the position
                for keypoint_pair in data_format.landmarks.asymm_pairs:
                    keypoint_pair = np.array(keypoint_pair)
                    landmarks[:, keypoint_pair,
                              :] = landmarks[:, keypoint_pair[::-1], :]
                    # Convert back to original format
                landmarks = landmarks.reshape((-1, n_keypoints * 2))
        np.put_along_axis(ret_targets, values=landmarks, axis=lmk_axis,
                          indices=np.array(lmk_indices)[np.newaxis, :])
    return ret_targets


def create_transform(transforms: List = None, *args, **kwargs):
    if transforms is None:
        raise KeyError(
            "Albumentations module args expecting 'transforms' as one of the config args!")
    else:
        if not isinstance(transforms, list):
            raise TypeError(
                "Albumentations 'transforms' args expecting list as the input type, got %s" % type(transforms))
    albumentations_wrapper = AlbumentationsWrapper(
        transforms=transforms, *args, **kwargs)
    return albumentations_wrapper
