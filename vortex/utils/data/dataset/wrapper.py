from pathlib import Path
from easydict import EasyDict
from collections import namedtuple
from multipledispatch import dispatch
from typing import Dict, List, Union, Callable, Tuple

import cv2
import torch
import random
import numpy as np
# import torchvision.transforms.functional as tf
from vortex.networks.modules.preprocess.normalizer import to_tensor,normalize
import albumentations.core.composition as albumentations_compose
import albumentations.augmentations.transforms as albumentations_tf
from PIL.Image import Image
import PIL
import warnings

import os
import sys

from ..augment import create_transform
from .dataset import get_base_dataset

KNOWN_DATA_FORMAT = ['class_label', 'bounding_box', 'landmarks']


class DatasetWrapper:
    """ Intermediate wrapper for external dataset.

    This module is used to build external dataset and act as a dataset iterator
    which also applies preprocess stages for the output of external dataset.

    The image input will automatically be resized to desired `input_size` from
    `preprocess_args` retrieved from config.

    Args:
        dataset (str): external dataset name to be built. see available (###input context here).
        stage (str): stages in which the dataset be used, valid: `train` and `validate`.
        preprocess_args (EasyDict): pre-process options for input image from config, see (###input context here).
        augments (sequence, or callable): augmentations to be applied to the output of external dataset, see (###input context here).
        annotation_name (EasyDict): (###unused), see (###input context here).
    """

    def __init__(self, dataset: str, stage: str, preprocess_args: Union[EasyDict, dict],
                 augmentations: Union[Tuple[str, dict], List, Callable] = None,
                 dataset_args: Union[EasyDict, dict] = {}, annotation_name='bboxes'):

        self.stage = stage
        self.preprocess_args = preprocess_args
        self.dataset = get_base_dataset(
            dataset, dataset_args=dataset_args)
        if not 'data_format' in self.dataset.__dict__.keys():
            raise RuntimeError("expects dataset `%s` to have `data_format` field : dict<str,dict>, explaining data format (e.g. bounding_box, class_label, landmarks etc)" % dataset)
        if not 'class_names' in self.dataset.__dict__.keys():
            raise RuntimeError("expects dataset `%s` to have `class_names` field : list<str>, explaining class string names which also map to its index positioning in the list. E.g. self.class_names = ['cat','dog'] means class_label = 0 is 'cat' " % dataset)
        self.data_format = EasyDict(self.dataset.data_format)
        # Data format standard check
        self.data_format = check_data_format_standard(self.data_format)

        # Configured computer vision augmentation initialization
        self.augments = None
        if stage == 'train' and augmentations is not None:
            self.augments = []
            if not isinstance(augmentations, List):
                raise TypeError('expect augmentations config type as a list, got %s' % type(augmentations))
            for augment in augmentations:
                module_name = augment.module
                module_args = augment.args
                if not isinstance(module_args, dict):
                    raise TypeError("expect augmentation module's args value to be dictionary, got %s" % type(module_args))
                tf_kwargs = module_args
                tf_kwargs['data_format'] = self.data_format
                augments = create_transform(module_name, **tf_kwargs)
                self.augments.append(augments)
        # Standardized computer vision augmentation initialization, longest resize and pad to square
        standard_tf_kwargs = EasyDict()
        standard_tf_kwargs.data_format = self.data_format
        standard_tf_kwargs.transforms = [
            {'transform': 'LongestMaxSize', 'args': {'max_size': preprocess_args.input_size}},
            {'transform': 'PadIfNeeded', 'args': {'min_height': preprocess_args.input_size,
                                                  'min_width': preprocess_args.input_size,
                                                  'border_mode': cv2.BORDER_CONSTANT,
                                                  'value': [0, 0, 0]}},
        ]
        self.standard_augments = create_transform(
            'albumentations', **standard_tf_kwargs)

        self.preprocess_args = preprocess_args
        self.annotation_name = annotation_name
        if self.stage == 'train':
            assert hasattr(self.preprocess_args, "input_size"), "at stage 'train', 'input_size' is mandatory for preprocess, please specify 'input_size' at 'preprocess_args'"
            assert hasattr(self.preprocess_args, "input_normalization"), "at stage 'train', 'input_normalization' is mandatory for preprocess, please specify 'input_normalization' at 'preprocess_args'"
            assert all([key in self.preprocess_args.input_normalization.keys() for key in ['std', 'mean']]), "at stage 'train', please specify 'mean' and 'std' for 'input_normalization'"
            assert isinstance(self.preprocess_args.input_size, int)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, target = self.dataset[index]
        # Currently support decoding image file provided it's string path using OpenCV (BGR format), for future roadmap if using another decoder
        if isinstance(image, str):
            if not Path(image).is_file():
                raise RuntimeError("Image file at '%s' not found!! Please check!" % (image))
            image = cv2.imread(image)

        # If dataset is PIL Image, convert to numpy array, support for torchvision dataset
        elif isinstance(image, PIL.Image.Image):
            image = np.array(image)

        # From this point, until specified otherwise, image is in numpy array format
        elif isinstance(image, np.ndarray):
            pass
        else:
            raise RuntimeError("Unknown return format of %s" % type(image))

        # Support if provided target only an int, convert to np.array
        # Classification case doesn't need any array slicing so constructed np array only array of 1 is enough
        if isinstance(target, int):
            target = np.array([target])

            if self.dataset.data_format['class_label'] is not None:
                warnings.warn("'int' type target should be paired with 'class_label' data_format with value None. Updating config..")
                self.dataset.data_format['class_label']=None

        # Target must be numpy array
        if not isinstance(target, np.ndarray):
            raise RuntimeError("Unknown target return of %s" % type(target))

        # Configured computer vision augment -- START
        if self.stage == 'train' and self.augments is not None:
            # Out of image shape coordinates adaptation -- START
            image, target = check_and_fix_coordinates(
                image, target, self.data_format)
            # Out of image shape coordinates adaptation --  END
            for augment in self.augments:
                image, target = augment(image, target)
                if target.shape[0] < 1:
                    raise RuntimeError("The configured augmentations resulting in 0 shape target!! Please check your augmentation and avoid this!!")
        if not isinstance(image, PIL.Image.Image) and not isinstance(image, np.ndarray):
            raise RuntimeError('Expected augmentation output in PIL.Image.Image or numpy.ndarray format, got %s ' % type(image))
        if (np.all(image >= 0.) and np.all(image <= 1.)) or isinstance(image, torch.Tensor):
            pixel_min, pixel_max = np.min(image), np.max(image)
            raise RuntimeError('Augmentation image output expect unnormalized pixel value (0-255), got min %2.2f and max %2.2f' % (pixel_min, pixel_max))
        # Configured computer vision augment -- END
        # Standard computer vision augment -- START
        image, target = self.standard_augments(image, target)
        if self.stage == 'train':
            input_normalization = self.preprocess_args.input_normalization
            if 'scaler' not in input_normalization:
                input_normalization.scaler=255
            image = torch.from_numpy(np.expand_dims(image, axis=0))
            image = to_tensor(image,scaler=input_normalization.scaler)
            image = normalize(
                image, input_normalization.mean, input_normalization.std).squeeze(0)
        # Standard computer vision augment -- END
        if not isinstance(target, torch.Tensor):
            if isinstance(target, np.ndarray):
                target = torch.from_numpy(target)
            else:
                raise RuntimeError(
                    'unsupported data type for target, got %s' % type(target))

        data = image, target
        return data


def check_data_format_standard(data_format: EasyDict):
    def _convert_indices_to_list(indices: EasyDict):
        '''
        Convert 'indices' in dictionary dict with 'start' and 'end' keys to list of int
        '''
        converted_indices = [*range(indices.start, indices.end+1)]
        return converted_indices
    if data_format:
        # Check if data_format keys is acceptable
        if not all([key in KNOWN_DATA_FORMAT for key in data_format]):
            raise RuntimeError("Unknown data format found! Please check! Known data format = %s, found %s" % (
                KNOWN_DATA_FORMAT, data_format.keys()))
        for key in data_format:
            # Check if data_format key value is None other than 'class_label'
            if data_format[key] is None and key != 'class_label':
                raise RuntimeError("Only 'class_label' data format can contain None value, found '%s' contain None" % key)
            if data_format[key] is not None:
                # Check if 'indices' exist in each data_format
                if 'indices' not in data_format[key]:
                    raise RuntimeError("'indices' key not found in '%s', please check!, found %s" % (key, data_format[key].keys()))
                # Check if data_format 'indices' type must be a list or a dict
                if type(data_format[key].indices) not in [list, EasyDict]:
                    raise RuntimeError("Expected '%s' 'indices' value type as list or dict, found %s" % (key, type(
                        data_format[key].indices)))
                # If 'indices' is dict, check for 'start' and 'end' keys
                if isinstance(data_format[key].indices, EasyDict):
                    # Convert 'indices' to list of int
                    if 'start' in data_format[key].indices and 'end' in data_format[key].indices:
                        data_format[key].indices = _convert_indices_to_list(
                            data_format[key].indices)
                    else:
                        raise RuntimeError("Expected '%s' 'indices' in dictionary type contain 'start' and 'end' keys, found %s please check!!" % (key, data_format[key].indices.keys()))
            # Bounding box data format standard checking
            if key == 'bounding_box':
                # Assume one class object detection if 'class_label' not found
                if 'class_label' not in data_format:
                    data_format.class_label = None
                # Check 'indices' length, must exactly 4 represent x,y,w,h
                if len(data_format.bounding_box.indices) != 4:
                    raise RuntimeError("'bounding_box' 'indices' data format must have length of 4!, found %s length of %i" % (
                        data_format.bounding_box.indices, len(data_format.bounding_box.indices)))

            # Landmarks data format standard checking
            if key == 'landmarks':
                # Check 'indices' length, must be in even number
                if len(data_format.landmarks.indices) % 2 != 0:
                    raise RuntimeError("'landmarks' 'indices' length must be an even number to be sliced equally into x,y coords! Found length of the 'indices' equal to %i ! Please check !" % (
                        len(data_format.landmarks.indices)))
                # Check 'asymm_pairs' format
                if 'asymm_pairs' in data_format.landmarks:
                    # Check type
                    if not isinstance(data_format.landmarks.asymm_pairs, list):
                        raise RuntimeError("'landmarks' 'asymm_pairs' value type must be a list!, found %s" % type(
                            data_format.landmarks.asymm_pairs))
                    # Check if 'asymm_pairs' contain empty list
                    if len(data_format.landmarks.asymm_pairs) == 0:
                        raise RuntimeError("'landmarks' 'asymm_pairs' contain 0 pairs! Please check!")
                    # Check pairs type inside the list
                    for pair in data_format.landmarks.asymm_pairs:
                        if not isinstance(pair, list):
                            raise RuntimeError("'landmarks' 'asymm_pairs' pairs value type inside the list must be a list!, found %s" % type(pair))
                        if len(pair) != 2 or not all([isinstance(value, int) for value in pair]):
                            raise RuntimeError("'landmarks' 'asymm_pairs' pairs value type inside the list must be list of int with the length of 2!, found %s" % pair)

        return data_format
    else:
        raise RuntimeError('Empty data_format dictionary! Please check!')


def check_and_fix_coordinates(image: np.ndarray, target: np.ndarray, data_format: EasyDict):
    # Check bounding box coordinates
    if 'bounding_box' in data_format:
        box_indices, box_axis = data_format.bounding_box.indices, data_format.bounding_box.axis
        # # VIZ DEBUG
        # ori_vis_image = image.copy()
        # h, w, c = ori_vis_image.shape
        # allbboxes = np.take(
        #     target, axis=box_axis, indices=box_indices)
        # for bbox in allbboxes:
        #     x = int(bbox[0]*w)
        #     y = int(bbox[1]*h)
        #     box_w = int(bbox[2]*w)
        #     box_h = int(bbox[3]*h)
        #     cv2.rectangle(ori_vis_image, (x, y),
        #                   (x+box_w, y+box_h), (0, 0, 255), 2)
        # cv2.imshow('ori', ori_vis_image)
        # # VIZ DEBUG

        # Slice all xy min coords
        allbboxes_xy = np.take(target, axis=box_axis, indices=box_indices[0:2])
        # Evaluate all xy min coords less than 0 (relative coords) and update value to 0
        allbboxes_xy[np.where(allbboxes_xy < 0)] = 0
        # Slice all wh and calculate xymax
        allbboxes_wh = np.take(target, axis=box_axis, indices=box_indices[2:4])
        allbboxes_xymax = allbboxes_xy+allbboxes_wh
        # Evaluate all xy max coords more than 0.99 (relative coords) and update value to 0.99
        # Why not 1? Risk of overflow even 1e-5 more than 1 cause albumentations error
        allbboxes_xymax[np.where(allbboxes_xymax > 1)] = 1
        # Get new wh after update
        allbboxes_wh = allbboxes_xymax-allbboxes_xy

        # Update target tensor
        np.put_along_axis(target, values=allbboxes_xy, indices=np.array(box_indices[0:2])[np.newaxis, :], axis=box_axis)
        np.put_along_axis(target, values=allbboxes_wh, indices=np.array(box_indices[2:4])[np.newaxis, :], axis=box_axis)
        target = target.astype('float32')
        # # VIZ DEBUG
        # fixed_vis_image = image.copy()
        # allbboxes = np.take(
        #     target, axis=box_axis, indices=box_indices)
        # for bbox in allbboxes:
        #     x = int(bbox[0]*w)
        #     y = int(bbox[1]*h)
        #     box_w = int(bbox[2]*w)
        #     box_h = int(bbox[3]*h)
        #     cv2.rectangle(fixed_vis_image, (x, y),
        #                   (x+box_w, y+box_h), (0, 0, 255), 2)
        # cv2.imshow('fixed', fixed_vis_image)
        # cv2.waitKey(0)
        # # VIZ DEBUG

    if 'landmarks' in data_format:
        # Get image shape
        img_h, img_w, c = image.shape
        # Slice landmarks tensor
        landmarks_indices, landmarks_axis = data_format.landmarks.indices, data_format.landmarks.axis
        landmarks_tensor = np.take(target, axis=landmarks_axis,indices=landmarks_indices
        )
        # Prepare bounding_box_modification
        if 'bounding_box' in data_format:
            box_indices, box_axis = data_format.bounding_box.indices, data_format.bounding_box.axis
            allbboxes = np.take(target, axis=box_axis, indices=box_indices)
            # Convert xywh to abs coords
            allbboxes[:, [box_indices[0], box_indices[2]]] *= img_w
            allbboxes[:, [box_indices[1], box_indices[3]]] *= img_h

        # Get x,y slice
        n_pairs = len(landmarks_tensor[0]) // 2
        landmarks_x = landmarks_tensor[:, 0:n_pairs*2:2]
        landmarks_y = landmarks_tensor[:, 1:n_pairs*2:2]

        # Convert to absolute value
        abs_landmarks_x = (landmarks_x*img_w).astype('int')
        abs_landmarks_y = (landmarks_y*img_h).astype('int')

        # # VIZ DEBUG
        # ori_vis_image = image.copy()
        # for bbox in allbboxes:
        #     x = int(bbox[0])
        #     y = int(bbox[1])
        #     box_w = int(bbox[2])
        #     box_h = int(bbox[3])
        #     cv2.rectangle(ori_vis_image, (x, y),
        #                   (x+box_w, y+box_h), (0, 0, 255), 2)
        # for j, landmark in enumerate(abs_landmarks_x):
        #     for i, landmark_x in enumerate(landmark):
        #         x = int(landmark_x)
        #         y = int(abs_landmarks_y[j][i])
        #         cv2.circle(ori_vis_image, (x, y),
        #                    2, (0, 0, 255), -1)
        # # VIZ DEBUG

        # Get minimum and maximum coordinates both on x and y
        min_x = np.min(abs_landmarks_x)
        max_x = np.max(abs_landmarks_x)
        min_y = np.min(abs_landmarks_y)
        max_y = np.max(abs_landmarks_y)

        # Calculate padded border to accomodate out of bonds landmarks
        left_border = 0
        right_border = 0
        top_border = 0
        bottom_border = 0
        modify = False
        if min_x < 0:
            left_border = int(0-min_x)
            abs_landmarks_x += left_border
            if 'bounding_box' in data_format:
                allbboxes[:, box_indices[0]] += left_border
            modify = True
        if min_y < 0:
            top_border = int(0-min_y)
            abs_landmarks_y += top_border
            if 'bounding_box' in data_format:
                allbboxes[:, box_indices[1]] += top_border
            modify = True
        if max_x > img_w:
            right_border = int(max_x-img_w)
            modify = True
        if max_y > img_h:
            bottom_border = int(max_y-img_h)
            modify = True

        # Modify if any modification is needed
        if modify:
            # Pad image
            pad_color = [0, 0, 0]
            image = cv2.copyMakeBorder(image, top_border, bottom_border, left_border, right_border, cv2.BORDER_CONSTANT,value=pad_color)
            new_img_h, new_img_w, c = image.shape
            # Modify bounding boxes annotations if any
            if 'bounding_box' in data_format:
                allbboxes[:, [box_indices[0], box_indices[2]]] /= new_img_w
                allbboxes[:, [box_indices[1], box_indices[3]]] /= new_img_h
                np.put_along_axis(target, values=allbboxes, indices=np.array(box_indices)[np.newaxis, :], axis=box_axis)
            # Modify landmarks tensor
            landmarks_x = abs_landmarks_x / new_img_w
            landmarks_y = abs_landmarks_y / new_img_h

            landmarks_tensor[:, 0:n_pairs*2:2] = landmarks_x
            landmarks_tensor[:, 1:n_pairs*2:2] = landmarks_y
            np.put_along_axis(target, values=landmarks_tensor, indices=np.array(landmarks_indices)[np.newaxis, :], axis=landmarks_axis)

            # # VIZ DEBUG
            # cv2.imshow('ori', ori_vis_image)
            # fix_vis_image = image.copy()
            # box_indices, box_axis = data_format.bounding_box.indices, data_format.bounding_box.axis
            # allbboxes = np.take(
            #     target, axis=box_axis, indices=box_indices)
            # allbboxes[:, [box_indices[0], box_indices[2]]] *= new_img_w
            # allbboxes[:, [box_indices[1], box_indices[3]]] *= new_img_h
            # landmarks_indices, landmarks_axis = data_format.landmarks.indices, data_format.landmarks.axis
            # landmarks_tensor = np.take(
            #     target, axis=landmarks_axis,
            #     indices=landmarks_indices
            # )
            # n_pairs = len(landmarks_tensor[0]) // 2
            # landmarks_x = landmarks_tensor[:, 0:n_pairs*2:2]
            # landmarks_y = landmarks_tensor[:, 1:n_pairs*2:2]
            # abs_landmarks_x = (landmarks_x*new_img_w).astype('int')
            # abs_landmarks_y = (landmarks_y*new_img_h).astype('int')
            # for bbox in allbboxes:
            #     x = int(bbox[0])
            #     y = int(bbox[1])
            #     box_w = int(bbox[2])
            #     box_h = int(bbox[3])
            #     cv2.rectangle(fix_vis_image, (x, y),
            #                   (x+box_w, y+box_h), (0, 0, 255), 2)
            # for j, landmark in enumerate(abs_landmarks_x):
            #     for i, landmark_x in enumerate(landmark):
            #         x = int(landmark_x)
            #         y = int(abs_landmarks_y[j][i])
            #         cv2.circle(fix_vis_image, (x, y),
            #                    2, (0, 0, 255), -1)
            # cv2.imshow('fix', fix_vis_image)
            # cv2.waitKey(0)
            # # VIZ DEBUG
    return image, target


# def create_dataset(dataset: str, stage: str, preprocess_args=None,
#                    dataset_args=None, augmentations=None):
#     """ Create external dataset with `DatasetWrapper` module.

#     Args:
#         dataset (str): external dataset name, see available (###input context here).
#         stage (str): stages in which the dataset be used, valid: `train` and `validation`.
#         dataset_args (dict): additional argument for dataset object building.
#         augmentations: augmentation config.
#         preprocess_args: preprocess arguments from config.

#     Returns:
#         DatasetWrapper: external dataset that built with `DatasetWrapper` module.
#     """

#     return DatasetWrapper(dataset=dataset, stage=stage, preprocess_args=preprocess_args,
#                           augmentations=augmentations, dataset_args=dataset_args)
