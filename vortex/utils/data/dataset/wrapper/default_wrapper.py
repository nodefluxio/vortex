from pathlib import Path
from easydict import EasyDict
from typing import Dict, List, Union, Callable, Tuple

import cv2
import torch
import numpy as np
# import torchvision.transforms.functional as tf
from vortex.networks.modules.preprocess.normalizer import to_tensor,normalize
import albumentations.core.composition as albumentations_compose
import albumentations.augmentations.transforms as albumentations_tf
from PIL.Image import Image
import PIL
import warnings

from ...augment import create_transform
from .base_wrapper import BaseDatasetWrapper,check_and_fix_coordinates

class DefaultDatasetWrapper(BaseDatasetWrapper):
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
                 dataset_args: Union[EasyDict, dict] = {}):
        super().__init__(dataset=dataset,
                         stage=stage,
                         preprocess_args=preprocess_args,
                         augmentations=augmentations,
                         dataset_args=dataset_args
                         )
        # Configured computer vision augmentation initialization
        self.augments = None
        if stage == 'train' and self.augmentations_list is not None:
            self.augments = []
            if not isinstance(self.augmentations_list, List):
                raise TypeError('expect augmentations config type as a list, got %s' % type(self.augmentations_list))
            for augment in self.augmentations_list:
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
            if pixel_min == 0. and pixel_max == 0.:
                raise RuntimeError('Augmentation image output producing blank image ( all pixel value == 0 ), please check and visualize your augmentation process!!')
            else:
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