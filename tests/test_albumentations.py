import unittest
from easydict import EasyDict
import numpy as np

from vortex.utils.data.augment import create_transform
from vortex.utils.data.dataset.wrapper.basic_wrapper import check_data_format_standard

class TestAugmentation():
    def test_albumentations(self):
        data_format = EasyDict({
            'bounding_box': {
                'indices': [0, 1, 2, 3],
                'axis': 1,
            },
            'class_label': None,
            'landmarks': {
                'indices': {
                    'start': 4,
                    'end': 13
                },
                'asymm_pairs': [[0, 1], [3, 4]],
                'axis': 1
            }})
        data_format = check_data_format_standard(data_format)
        transform_args = EasyDict({
            'transforms': [
                {'transform': 'HorizontalFlip', 'args': {'p': 1}}
            ],
            'bbox_params': {'min_area': 0.0, 'min_visibility': 0.0},
            'visual_debug': False,
        })
        transform_args.data_format = data_format

        test_image = np.random.rand(350, 350, 3)
        test_targets = np.array(
            [[0.5, 0.3, 0.1, 0.2, 132.163, 88.392, 163.809, 86.472, 164.224, 113.744, 129.294, 131.679, 158.74, 131.098],
             [0.5, 0.3, 0.1, 0.2, 132.163, 88.392, 163.809, 86.472,
                 164.224, 113.744, 129.294, 131.679, 158.74, 131.098]
             ]
        )

        albumentations_wrapper = create_transform(
            'albumentations', **transform_args)

        image, targets = albumentations_wrapper(test_image, test_targets)
        # Check output image type
        assert isinstance(image, np.ndarray)
        # Check output targets type
        assert isinstance(targets, np.ndarray)
        # Check output image channels
        assert image.shape[-1] == 3
        # Check output target annotations length must be same as input
        assert test_targets.shape[-1] == targets.shape[-1]
