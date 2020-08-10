import os
import os.path
import sys

import cv2
import random
import logging
import numpy as np
from pathlib import Path
from easydict import EasyDict
from typing import List, Union, Tuple, Callable


class DarknetCollate:
    """
    A collater returning darknet format
    """

    def __init__(self, dataformat: dict):
        # TODO : check all necessary fields
        self.dataformat = EasyDict(dataformat)

    def __call__(self, batch):
        try:
            import torch
        except:
            raise RuntimeError("current implementation needs torch")
        images, targets = list(zip(*batch))
        images = torch.stack(images)
        collate_targets = []

        df = self.dataformat
        for batch_idx, target in enumerate(targets):
            if not len(target.shape) == 2:
                raise RuntimeError(
                    "expects dimensionality of target is 2 got %s" % len(target.shape))
            if self.dataformat.class_label is None:
                class_label_tensor = torch.from_numpy(
                    np.array([[0]]*target.shape[0])).float()
            else:
                class_label_tensor = torch.index_select(
                    input=target,
                    dim=df.class_label.axis,
                    index=torch.tensor(
                        df.class_label.indices, dtype=torch.long)
                )
            img_idx_tensor = torch.from_numpy(
                np.array([[batch_idx]]*target.shape[0])).float()
            bounding_box_tensor = torch.index_select(
                input=target,
                dim=df.bounding_box.axis,
                index=torch.tensor(df.bounding_box.indices, dtype=torch.long)
            ).float()
            target = torch.cat(
                (class_label_tensor, bounding_box_tensor), axis=1)
            target = torch.cat((img_idx_tensor, target), axis=1)
            collate_targets.append(target)

        targets = torch.cat(collate_targets)
        return (images, targets)


class MultiScaleDarknetCollate:
    """
    Darknet collater with support for multiscale training
    """

    def __init__(self, scales: Union[List[int], int], dataformat: dict):
        if isinstance(scales, int):
            scales = [scales]
        self.scales = scales
        # TODO : check all necessary fields
        self.dataformat = EasyDict(dataformat)

    def __call__(self, batch):
        try:
            import torch
            import torch.nn.functional as F
        except:
            raise RuntimeError("current implementation needs torch")
        img_size = self.scales[random.randrange(0, len(self.scales))]
        images, targets = list(zip(*batch))
        images = torch.stack(images)
        images = F.interpolate(images, size=img_size)
        collate_targets = []

        df = self.dataformat
        for batch_idx, target in enumerate(targets):
            if not len(target.shape) == 2:
                raise RuntimeError(
                    "expects dimensionality of target is 2 got %s" % len(target.shape))
            if self.dataformat.class_label is None:
                class_label = torch.from_numpy(
                    np.array([[0]]*target.shape[0])).float()
            else:
                class_label = torch.index_select(
                    input=target,
                    dim=df.class_label.axis,
                    index=torch.tensor(
                        df.class_label.indices, dtype=torch.long)
                )
            idx_tensor = torch.from_numpy(
                np.array([[batch_idx]]*target.shape[0])).float()
            target = torch.index_select(
                input=target,
                dim=df.bounding_box.axis,
                index=torch.tensor(df.bounding_box.indices, dtype=torch.long)
            )
            target = torch.cat((class_label, target), axis=1)
            target = torch.cat((idx_tensor, target), axis=1)
            collate_targets.append(target)

        targets = torch.cat(collate_targets)
        return (images, targets)


supported_collater = [
    'DarknetCollate',
    'MultiScaleDarknetCollate',
]


def create_collater(collater: str, *args, **kwargs):
    if not collater in supported_collater:
        raise RuntimeError("unsupported collater %s, supported : %s" % (
            collater, ','.join(supported_collater)))
    return eval(collater)(*args, **kwargs)
